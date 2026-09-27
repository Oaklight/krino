"""Generate LLM-based question-specific confounders for augmented noul items.

Reads noul questions from families JSONL files, generates 3 question-specific
confounder options via an LLM (Qwen3.8-27B or similar), and writes results to
separate `{domain}_llm_confounders.jsonl` files (one record per noul question).
Does NOT modify families in-place — use --merge to apply confounders back.

Resumable: skips questions whose IDs already appear in the output file.

Usage:
    python generate_llm_confounders.py --data-dir /path/to/synthetic/    # dry-run
    python generate_llm_confounders.py --data-dir /path/to/synthetic/ --apply
    python generate_llm_confounders.py --data-dir /path/to/synthetic/ --apply --shard 0 --num-shards 2
    python generate_llm_confounders.py --data-dir /path/to/synthetic/ --merge   # apply to families
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger("generate_llm_confounders")

_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_BASE_URL = "http://localhost:8765"


def _build_augment_prompt(
    state: str, instructions: str, label: bool, cognitive_type: str
) -> str:
    return (
        f"You are expanding a binary (true/false) decision question into a "
        f"multi-option question by generating plausible alternative answers.\n\n"
        f"Context:\n"
        f"  State: {state[:800]}\n"
        f"  Question: {instructions}\n"
        f"  Correct answer: {'true' if label else 'false'}\n"
        f"  Cognitive type: {cognitive_type}\n\n"
        f"Generate exactly 3 additional answer options that are:\n"
        f"- Plausible but distinguishably different from true/false\n"
        f"- Semantically relevant to the question domain\n"
        f"- Meaningful gradations (e.g. partial, conditional, uncertain)\n\n"
        f"Rules:\n"
        f"- Keys must be snake_case identifiers\n"
        f"- Do NOT include 'true' or 'false' as keys\n"
        f"- Each description should be 5-15 words\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"additional_options": {{"<key1>": "<description>", "<key2>": "<description>", "<key3>": "<description>"}}}}'
    )


def _parse_augment_response(content: str) -> dict[str, str] | None:
    text = _THINK_RE.sub("", content).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    additional = parsed.get("additional_options")
    if not isinstance(additional, dict):
        return None

    filtered = {
        k: str(v)
        for k, v in additional.items()
        if isinstance(k, str)
        and _SNAKE_RE.match(k)
        and k not in ("true", "false")
        and isinstance(v, str)
        and v.strip()
    }

    if len(filtered) < 2:
        return None

    base = {
        "true": "The stated condition is confirmed by the available evidence",
        "false": "The stated condition is contradicted by the available evidence",
    }
    base.update(filtered)
    return base


def _discover_domains(data_dir: Path) -> list[str]:
    return sorted(
        f.name.replace("_families.jsonl", "")
        for f in data_dir.glob("*_families.jsonl")
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(items: list[dict], path: Path) -> None:
    with open(path, "a") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_jsonl(items: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def shard_domains(domains: list[str], shard: int, num_shards: int) -> list[str]:
    return [d for i, d in enumerate(sorted(domains)) if i % num_shards == shard]


def _confounder_path(data_dir: Path, domain: str) -> Path:
    return data_dir / f"{domain}_llm_confounders.jsonl"


async def _generate_confounders_for_domain(
    domain: str,
    data_dir: Path,
    model: str,
    base_url: str,
    max_concurrent: int,
) -> tuple[int, int, int]:
    """Generate LLM confounders for one domain. Returns (skipped, generated, failed)."""
    import urllib.request

    fam_path = data_dir / f"{domain}_families.jsonl"
    out_path = _confounder_path(data_dir, domain)
    families = _load_jsonl(fam_path)

    existing_ids: set[str] = set()
    for rec in _load_jsonl(out_path):
        existing_ids.add(rec.get("id", ""))

    work: list[tuple[int, int, dict, str]] = []
    skipped = 0
    for fi, fam in enumerate(families):
        state = fam.get("state", "")
        for ni, nq in enumerate(fam.get("noul_questions", [])):
            rec_id = f"{domain}-{fi:04d}-{ni}"
            if rec_id in existing_ids:
                skipped += 1
                continue
            work.append((fi, ni, nq, state))

    if not work:
        return skipped, 0, 0

    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[dict | None] = [None] * len(work)

    async def _do_one(idx: int, fi: int, ni: int, nq: dict, state: str) -> None:
        prompt = _build_augment_prompt(
            state,
            nq.get("instructions", ""),
            nq.get("label", True),
            nq.get("cognitive_type", "unknown"),
        )
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()

        for attempt in range(3):
            try:
                async with semaphore:
                    loop = asyncio.get_event_loop()
                    req = urllib.request.Request(
                        f"{base_url}/v1/chat/completions",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    resp_data = await loop.run_in_executor(
                        None,
                        lambda: urllib.request.urlopen(req, timeout=60).read(),
                    )
                resp = json.loads(resp_data)
                content = resp["choices"][0]["message"]["content"]
                options = _parse_augment_response(content)
                if options:
                    results[idx] = {
                        "id": f"{domain}-{fi:04d}-{ni}",
                        "family_idx": fi,
                        "nq_idx": ni,
                        "cognitive_type": nq.get("cognitive_type", "unknown"),
                        "augmented_options_llm": options,
                        "model": model,
                    }
                    return
                else:
                    logger.warning(
                        "  Parse fail %s fi=%d ni=%d (attempt %d): %s",
                        domain, fi, ni, attempt + 1, content[:100],
                    )
            except Exception as exc:
                logger.warning(
                    "  API error %s fi=%d ni=%d (attempt %d): %s",
                    domain, fi, ni, attempt + 1, exc,
                )
                await asyncio.sleep(2**attempt)

    tasks = [_do_one(i, fi, ni, nq, state) for i, (fi, ni, nq, state) in enumerate(work)]
    done = 0
    total = len(tasks)
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % 50 == 0 or done == total:
            ok = sum(1 for r in results if r is not None)
            logger.info("  %s: %d/%d done (%d ok)", domain, done, total, ok)

    new_records = [r for r in results if r is not None]
    if new_records:
        _append_jsonl(new_records, out_path)

    generated = len(new_records)
    failed = len(work) - generated
    return skipped, generated, failed


def _merge_confounders(data_dir: Path, domains: list[str]) -> None:
    """Merge LLM confounders back into families JSONL files."""
    for domain in sorted(domains):
        fam_path = data_dir / f"{domain}_families.jsonl"
        conf_path = _confounder_path(data_dir, domain)
        if not conf_path.exists():
            continue

        families = _load_jsonl(fam_path)
        confounders = _load_jsonl(conf_path)

        lookup: dict[tuple[int, int], dict] = {}
        for rec in confounders:
            lookup[(rec["family_idx"], rec["nq_idx"])] = rec

        applied = 0
        for fi, fam in enumerate(families):
            for ni, nq in enumerate(fam.get("noul_questions", [])):
                rec = lookup.get((fi, ni))
                if rec:
                    nq["augmented_options_llm"] = rec["augmented_options_llm"]
                    applied += 1

        if applied:
            _write_jsonl(families, fam_path)
            logger.info("  %s: merged %d/%d confounders", domain, applied, len(confounders))
        else:
            logger.info("  %s: nothing to merge", domain)


async def main_async(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)

    if args.domain:
        domains = [args.domain]
    else:
        domains = _discover_domains(data_dir)
        if args.shard is not None:
            domains = shard_domains(domains, args.shard, args.num_shards)
            logger.info("Shard %d/%d: %d domains", args.shard, args.num_shards, len(domains))

    if args.merge:
        logger.info("Merging LLM confounders into families...")
        _merge_confounders(data_dir, domains)
        return

    # Count work
    total_noul = 0
    total_need = 0
    for domain in domains:
        fam_path = data_dir / f"{domain}_families.jsonl"
        conf_path = _confounder_path(data_dir, domain)
        existing_ids: set[str] = set()
        for rec in _load_jsonl(conf_path):
            existing_ids.add(rec.get("id", ""))

        for fi, fam in enumerate(_load_jsonl(fam_path)):
            for ni, nq in enumerate(fam.get("noul_questions", [])):
                total_noul += 1
                if f"{domain}-{fi:04d}-{ni}" not in existing_ids:
                    total_need += 1

    logger.info("Domains: %d, Total noul: %d, Need LLM confounders: %d", len(domains), total_noul, total_need)

    if not total_need:
        logger.info("All noul questions already have LLM confounders.")
        return

    if not args.apply:
        logger.info("")
        logger.info("%-35s %8s %8s", "Domain", "Total", "Need")
        logger.info("-" * 55)
        for domain in sorted(domains):
            fam_path = data_dir / f"{domain}_families.jsonl"
            conf_path = _confounder_path(data_dir, domain)
            existing_ids = set()
            for rec in _load_jsonl(conf_path):
                existing_ids.add(rec.get("id", ""))
            n_total = 0
            n_need = 0
            for fi, fam in enumerate(_load_jsonl(fam_path)):
                for ni in range(len(fam.get("noul_questions", []))):
                    n_total += 1
                    if f"{domain}-{fi:04d}-{ni}" not in existing_ids:
                        n_need += 1
            if n_need:
                logger.info("  %-33s %8d %8d", domain, n_total, n_need)
        logger.info("-" * 55)
        logger.info("  %-33s %8d %8d", "TOTAL", total_noul, total_need)
        logger.info("\nDry-run mode. Use --apply to generate.")
        return

    grand_generated = 0
    grand_failed = 0
    t0 = time.monotonic()

    for domain in sorted(domains):
        logger.info("")
        logger.info("Processing %s...", domain)
        dt0 = time.monotonic()
        skipped, generated, failed = await _generate_confounders_for_domain(
            domain, data_dir, args.model, args.base_url, args.max_concurrent,
        )
        elapsed = time.monotonic() - dt0
        logger.info(
            "  %s: %d generated, %d skipped, %d failed (%.1fs)",
            domain, generated, skipped, failed, elapsed,
        )
        grand_generated += generated
        grand_failed += failed

    elapsed_total = time.monotonic() - t0
    logger.info("")
    logger.info("=" * 55)
    logger.info(
        "DONE: %d generated, %d failed (%.1fs total)",
        grand_generated, grand_failed, elapsed_total,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate LLM confounders for noul questions.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to synthetic data directory.")
    parser.add_argument("--apply", action="store_true", help="Actually call LLM API.")
    parser.add_argument("--merge", action="store_true", help="Merge confounder files back into families.")
    parser.add_argument("--domain", type=str, help="Process a single domain.")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0-based).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model identifier.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="vLLM base URL.")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent requests.")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
