import Link from "next/link";
import { loadDashboardData } from "@/lib/data";
import StatCard from "@/components/StatCard";

function pct(value: number | null | undefined): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function pctRange(min: number | null, max: number | null): string {
  if (min == null || max == null) return "—";
  return `${min.toFixed(1)}–${max.toFixed(1)}%`;
}

function ppRange(min: number | null, max: number | null): string {
  if (min == null || max == null) return "—";
  return `${min.toFixed(1)}–${max.toFixed(1)} pp`;
}

export default function HomePage() {
  const data = loadDashboardData();
  const landing = data.landing;
  const jvc = landing.judgement_vs_continuation;
  const band = landing.post_trained_4b;
  const resolution = landing.resolution;
  const banking = landing.banking77;

  const loss = jvc.losses[0];
  const bandNames = band.checkpoints.map((checkpoint) => checkpoint.model_id).join(", ");
  const topFourNames = resolution.top_four.map((checkpoint) => checkpoint.model_id).join(", ");
  const devPair = resolution.dev_pair.join(" vs ");

  return (
    <div className="max-w-7xl mx-auto px-4 py-12">
      {/* Hero */}
      <div className="mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-3">krino</h1>
        <p className="text-lg text-text-dim max-w-2xl">
          Typed decision models that output calibrated probabilities instead of
          free text. Open replication of TypeSafe&apos;s Jev.
        </p>
        <div className="mt-4 flex gap-3">
          <Link
            href="/benchmarks"
            className="px-4 py-2 rounded-[var(--radius)] bg-accent text-[var(--accent-on)] font-medium text-sm hover:opacity-90 transition-opacity"
          >
            View Benchmarks
          </Link>
          <a
            href="https://github.com/Oaklight/krino"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 rounded-[var(--radius)] border border-border text-sm hover:bg-bg-hover transition-colors"
          >
            GitHub
          </a>
        </div>
      </div>

      {/* Four claims, each with its source files */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
        <StatCard
          value={ppRange(jvc.above_10_min_pp, jvc.above_10_max_pp)}
          label="Judgement over continuation"
          detail={`${jvc.wins} of ${jvc.measured} checkpoints; ${jvc.above_10_count} win by more than 10 pp`}
          source="model/experiments/logit_prov/jevbench_{judgement,logit}_*.json; results/jevbench/logit_*.json"
        />
        <StatCard
          value={pctRange(band.min_pp, band.max_pp)}
          label="Post-trained 4B judgement"
          detail={`base ${pct(band.base_pp)}; ${bandNames}`}
          source={band.source_files.join("; ")}
        />
        <StatCard
          value={`${resolution.jevbench_pp.toFixed(1)} pp / ${resolution.dev_card_min_pp}–${resolution.dev_card_max_pp} pp`}
          label="Instrument resolution"
          detail="JevBench 231 / 812-row dev set at 80% power"
          source="model/evaluation/paired_power.py; model/experiments/resolution_inputs.json; data/benchmarks/dev/dev_800.json"
        />
        <StatCard
          value={String(landing.instrumented_rows)}
          label="Instrumented rows"
          detail="each names its readout and source file"
          source="web/public/results/eval_results.json"
        />
      </div>

      {/* The finding */}
      <section className="mb-12">
        <h2 className="text-xl font-semibold mb-4">What we have established</h2>
        <ul className="space-y-4 text-text-dim max-w-3xl">
          <li>
            On JevBench public, the per-option judgement readout beats option-text
            continuation on {jvc.wins} of {jvc.measured} checkpoints; on the{" "}
            {jvc.above_10_count} checkpoints where it wins by more than 10 pp the gap
            is {ppRange(jvc.above_10_min_pp, jvc.above_10_max_pp)}, largest on the{" "}
            <code className="font-mono text-text">{jvc.largest_tier}</code> tier.{" "}
            {loss
              ? `${loss.name} is the exception (${loss.gap_pp.toFixed(1)} pp).`
              : null}
          </li>
          <li>
            Post-training that moves the objective toward the readout helps: a 4B
            base scores {pct(band.base_pp)}, while the three-checkpoint 4B judgement
            band ({bandNames}) scores {pctRange(band.min_pp, band.max_pp)}.
          </li>
          <li>
            JevBench 231 cannot resolve differences below about{" "}
            {resolution.jevbench_pp.toFixed(1)} pp, which is why its top four
            checkpoints ({topFourNames}) are mutually indistinguishable within{" "}
            {resolution.top_four_spread_pp.toFixed(1)} pp. The 812-row development set
            resolves {resolution.dev_card_min_pp}–{resolution.dev_card_max_pp} pp and
            separates {devPair} ({Math.abs(resolution.dev_pair_diff_pp).toFixed(2)} pp,
            p = {resolution.dev_pair_p}) where JevBench has them{" "}
            {resolution.jevbench_pair_item_diff} item apart.
          </li>
        </ul>
      </section>

      {/* Banking77 moved out of the hero, labelled as in-domain */}
      <section className="mb-12 border-t border-border pt-8">
        <h2 className="text-xl font-semibold mb-3">In-domain single-task result</h2>
        <p className="text-text-dim max-w-3xl">
          Banking77, trained heads:{" "}
          <span className="font-mono text-text">{pct(banking.trained_pp)}</span>{" "}
          ({banking.trained_model}). Jev&apos;s reference is{" "}
          <span className="font-mono text-text">{pct(banking.jev_pp)}</span>. This is
          an in-domain single-task result; Jev&apos;s number is zero-shot across 77
          classes, so the comparison is not like-for-like.
        </p>
        <p className="mt-3 font-mono text-[10px] leading-snug text-text-dim break-all">
          {banking.trained_source_file}; {banking.jev_source_file}
        </p>
      </section>

      <p className="text-sm">
        <Link
          href="/benchmarks"
          className="text-accent hover:text-accent-hover transition-colors"
        >
          View the benchmarks tables →
        </Link>
      </p>
    </div>
  );
}
