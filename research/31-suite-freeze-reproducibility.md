# Retrospective: why the pinned Decision Index suite could not be reproduced later

**Status: 已完成.** This note answers one question: why a pinned suite hash that was
reproducible when it was made stopped being reproducible later, and what a freeze record
must contain so the same failure cannot recur. It is about the Decision Index kit, but
the rule applies to every suite this project pins, including the matched held-out.

## What happened

| date | event |
|---|---|
| 2026-09-19 | the kit freezes the 0.1 rows (seed 20260919) |
| 2026-09-22 | Decision Index 0.1 is published with rows sha256 `288d3720...` |
| 2026-10-13 | kit commit `2d7791c` fixes a future-turn leak in the SGD builder |
| 2026-11-06 | our clean rebuild produces `7068220a...`, not the pin |

The rebuilt counts are identical to the pin: 132,422 requests, 117,764 cases, 775,202
fields. So the selection did not move; the content of one benchmark did.

## What the evidence establishes

1. The pin **was** reproducible at the time it was made. Rebuilding SGD from the same
   pinned source with the pre-fix builder reproduces the pinned normalized hash
   `8496e556...` exactly. The pre-fix builder is deterministic.
2. The pin is **not** reproducible from the current kit. The current builder produces
   `e57d11fd...`, and the resulting v1 rows hash is `7068220a...`.
3. The difference is one correctness fix. Every SGD row used to hold a reference to a
   single growing `history` list, so each row carried the whole dialogue, future turns
   included. The fix copies the history per row.
4. Isolation is exact: 36 of 37 normalized files match the kit's pinned per-file hashes;
   only `SGD-service-given-intent.jsonl` differs. GSM8K v3, the one component that reads
   through pyarrow, matches its pin under a later pyarrow. Dependency drift is not the
   cause and pinning cannot fix it.

So the premise "the hash could not be reproduced from a clean checkout at the time" is
false. The failure is narrower and more useful: **the pin did not bind the code that
produced it.**

## The gap in the freeze record

The kit's `hub/manifest.json` records the output hashes, the seed, the creation time, the
source pins and the per-benchmark inventory. It does **not** record:

- the builder commit (`git rev-parse HEAD` of the kit);
- the exact dependency versions used to build (the package declares lower bounds only:
  `pyarrow>=20`, `pandas>=2.0`, `numpy>=1.26`, and so on);
- the Python version;
- a command that replays the build from a clean checkout.

`docs/suite.md` claims all 37 normalizers and the freeze step produced byte-identical
files. That claim was true when written and became false on 2026-10-13, when a builder
change landed without a suite version bump and without regenerating or re-pinning the
rows. Nothing in the kit rebuilds the frozen suite from a recorded tuple in CI, so the
claim was not checked after the change.

A hash is an identity, not a recipe. It can detect that content changed, but it cannot
say what produced the content, and it cannot distinguish "the sources drifted" from "the
builder changed" from "the dependency set changed".

## Why it was not caught at freeze time

At freeze time a clean rebuild from the then-current builder would have passed. The bug
was latent: the leaky output was the deterministic output of the leaky builder. The
failure only appears when the builder changes and the pin does not move. No amount of
care at the freeze would catch a change that had not happened yet.

What would have caught it is a **replay check after the change**, against a tuple that
includes the builder commit. When the fix landed, a CI job that rebuilt from the recorded
tuple would have failed and named the moved file. Without the builder commit in the
tuple, the same job would have rebuilt the corrected rows and compared them to a hash
that no longer described any code path in the repository — which is the state we found.

## Rules for every suite we pin

1. **The freeze tuple is the identity, not the hash alone.** Record:
   - suite version and the output hashes (whole file and per source file);
   - the builder commit;
   - an exact dependency lock (versions, not lower bounds);
   - the Python version and `PYTHONHASHSEED`, because an iteration order that reaches
     the output makes two otherwise identical rebuilds disagree;
   - the number of independent rebuilds compared, and whether they were byte-identical;
   - every upstream source revision;
   - the seed and the exact build command.
2. **Rebuild from the tuple in CI, per file.** Run it on every builder change and on a
   schedule. CI compares the per-file hashes, not only the whole-suite hash, so a failure
   names the moved benchmark. A mismatch stops the pipeline and is localized (point 5);
   it is never answered by a silent re-hash of the committed artifact. What to do about it
   is decided under point 4 and recorded before anything changes.
   The SGD case was localized in one command because the kit published per-file hashes.
   Without the automated comparison this rule is a reminder, and reminders do not survive.
3. **A builder change that alters row content bumps the suite version**, including a
   correctness fix. The trigger is not the intent of the change; it is the per-file replay
   in point 2 reporting a mismatch, and the decision under point 4 attributing it to the
   builder. Never regenerate rows under an existing version. The fix may make the old hash
   unreproducible on purpose; that is a reason to version, not to revert.
4. **Never relax the verifier or restore a defect to match a pin.** A mismatch means the
   tuple is wrong or the builder changed. Decide which first and record the decision
   before acting: if the tuple is wrong, correct the tuple; if the builder changed the
   content, bump the suite version under point 3 and update the tuple.
5. **Treat a mismatch as a localization task first.** Compare the per-file manifest before
   hypothesizing dependency or source drift. In this case the whole-suite hash pointed at
   132k rows; the per-file check pointed at one file in one command.
6. **Distinguish authoritative from correct.** A published pin is authoritative for
   comparability; it is not evidence that the content is valid. A suite with a future-turn
   leak is authoritative and wrong at the same time. Version the correction and say which
   identity a result used.

## What this project should adopt next

The matched held-out spec already records the corpus manifest hash, the input file hashes,
the builder commit, the LSH parameters and the assignment seed. It should add the
dependency lock, the Python version, `PYTHONHASHSEED`, the count of independent rebuilds
compared, and per-file output hashes, and the CI guard should rebuild from the tuple, not
only re-hash the committed file. The same tuple belongs in every future benchmark freeze,
including JevBench.

## Consequence for Decision Index

The corrected suite is `di-0.3-krino-1`: 140,620 requests, 140,178 scoreable, composite
`f725070c...`, rows `8224546d...`, added `7429f3c9...` (matches the pin), GSM8K
`75933489...` (matches the pin). It differs from the pinned edition only in the 2,500
selected SGD rows, where it removes the future-turn leak. Every gate row records the
suite version and hash; results are not compared across versions.
