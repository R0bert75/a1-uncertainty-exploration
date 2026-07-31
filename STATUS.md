# Project status — plain language

**As of 2026-07-30, `main` = `3f59fd7`, CI green, 335 tests.**

This document answers five questions in plain terms. It is a summary of the detailed
documents, not a replacement for them: the authoritative sources are
`audits/code_completeness_session3.md` (what is built), `protocol/FREEZE_READINESS.md` (what
blocks the freeze), and `protocol/preregistration.md` (what is frozen).

![Status against the implementation order]({{artifact:art_199027a0-5ea1-43e3-83bf-4ea26a86e1e1}})

---

## 1. Is the project well-thought, coherent, and in good shape?

**Yes on design. Yes on code. One real weakness in the paperwork.**

**The design is coherent.** The study asks one question — do uncertainty-aware exploration
methods beat a well-tuned ε-greedy Double DQN at low interaction budgets — and answers it
twice with complementary evidence. DeepSea gives exact ground truth (`Q*` is computable, so
"is this method's uncertainty *good*?" has a real answer rather than a proxy). MinAtar gives
external performance validity. Neither benchmark alone would support the claim; together
they do, and that pairing is the study's main structural strength.

**The rigor machinery is unusually strong** and is the project's real differentiator: a
pre-registration with a numbered freeze list, two decision gates, a cell-specific RNG
derivation scheme that makes runs byte-reproducible, an implementation-purity gate (C11) so
methods cannot differ by accident, and a positioning document that bans overclaiming
language. Most empirical RL work has none of this.

**Honest weaknesses:**

- **Two freeze-list values were never filled in** (below). They were supposed to be filled
  *before* the draft went to external review, so the reviewer is reading a document that is
  less complete than it claims to be. This is the one genuine process failure so far.
- **Scope is large.** 17 implementation steps, ~2,500–3,000 runs, two benchmark families,
  four methods, nine diagnostics. The descope ladder and compute caps exist precisely
  because this risk was recognized, but the risk is real.
- **No research results exist yet.** Everything so far is infrastructure. That is correct at
  this stage — the pre-registration forbids confirmatory runs before the freeze — but it
  means the project's scientific claims are entirely unvalidated so far.

---

## 2. What is the high-level plan?

Five phases:

1. **Build the machinery** (steps 1–9) — environments, four methods, logging, diagnostics,
   tuning. *We are here.*
2. **Freeze the protocol** — fill every freeze-list value, external methodological review,
   then tag it immutable. Nothing confirmatory may run before this.
3. **Pilot and tune** (steps 10–11) — measure effect sizes and wall-clock on development
   sizes, run the pre-registered hyperparameter searches. Feeds **Gate A**: is the study
   powered and affordable? If not, descend the descope ladder.
4. **Confirmatory block** (step 13) — run the frozen design once. Results are what they are;
   no re-tuning, no seed top-ups. Then **Gate B**, then the paper and public repo.
5. **Secondary work** (steps 15–17) — RQ3 analysis on existing runs, QR-DQN follow-up.

The whole point of the freeze-then-run structure is that phase 4 cannot be steered by its own
results. That is what makes the eventual claims credible, and it is why the paperwork in
phase 2 matters as much as the code.

---

## 3. Where are we, and is the work correct and robust?

**Position: end of the build phase, blocked before the freeze.** Steps 1–3 are done, 4–6 are
partly done, 7–9 are blocked on protocol values that were never written down.

**What runs today:** all four methods (DDQN, NoisyNet, BDQN, RP-BDQN) on both benchmark
families, from one shared code path; DeepSea's exact `Q*`; byte-reproducible seeding; CSV
logging on both reporting axes; figure generation; config-schema and purity audits.

**On correctness — the honest answer is "well-tested, not yet validated."** These are
different things and the distinction matters:

- **Well-tested: yes.** 335 tests. The strongest ones are not unit tests but *invariant*
  tests: re-running a config produces byte-identical output; the same replay data in
  `float32` and `uint8` storage produces identical batches; observations survive their dtype
  round-trip exactly across all six environments (2,000 steps each); a selection tie that
  can't be broken raises rather than silently resolving by input order.
- **Design quality: good, with reasoning recorded.** Decisions that could have been made
  casually were instead written down with their justification — why the conv trunk is a
  tunable nuisance rather than a frozen constant, why the diagnostics store raw samples
  rather than summaries, why the uint8 change is safe. Anyone can audit the reasoning, not
  just the code.
- **Not yet validated: also true.** No method has been shown to *learn well* — the only
  learning curve we have is a 30k-step smoke run, explicitly labeled exploratory. The code is
  correct in the sense of doing what it says; whether the DDQN baseline is *well-tuned* (which
  the entire comparison rests on) is unmeasured until the backbone tuning pass runs.

One caveat worth stating plainly: the tests were written by the same process that wrote the
code, so they check internal consistency and stated intent. They cannot catch a
misunderstanding shared between code and test. That is exactly what the external
methodological review is for.

---

## 4. What decisions are pending?

**Two are yours and block progress** (both recommendations were revised on 2026-07-30 — see the
audit note below):

| # | Decision | Recommendation | Why it blocks |
|---|---|---|---|
| 1 | Probe-set construction rule (freeze item 7) | **Exhaustive: `\|S\| = N(N+1)/2`, no cap.** No MinAtar probe set. | Item 7 is half-written. Blocks step 9 (the diagnostics battery). |
| 2 | Search distributions + tuning budget (freeze item 2) | **`n_backbone = 12`, `n_mini = 4`, 3 seeds per candidate** | Item 2 claims these are frozen; they exist nowhere. Blocks steps 5, 7, 8. |

Both recommendations, with the arithmetic behind them, are in
`protocol/decisions/staged_stage3_protocol_fixes.md`.

**Audit note — the earlier recommendations in this file were wrong.** An earlier version
recommended `|S| = 256` for DeepSea, `|S| = 512` for MinAtar, and `n_backbone = 24`. Checked
against the frozen documents rather than against intuition, all three failed:

- **`|S|` is not a value at all.** Item 7 says the `probe_set` stream governs *any* sampling —
  permissive, so enumeration is already compliant. Enumeration costs ~1.35 GB for the whole
  confirmatory sweep, while a 256 cap would have probed only 55% of reachable states at `N = 30`
  and 20% at `N = 50` — where RQ2-L is a v1.0 submission-gate deliverable. Enumeration also
  *removes* the item from the freeze list instead of filling it in.
- **MinAtar has no probe set.** Every §3.3 diagnostic references `Q*`, computable only on
  DeepSea; neither frozen document contains a MinAtar diagnostics clause. The 512 answered a
  question the protocol never asked.
- **`n_backbone = 24` overspends its budget by 40%.** Its justification (`192 + 32 + 12 = 236 ≈
  240`) was wrong three ways: the backbone is tuned *once* on DDQN, not per method; §3.4's 240
  counts runs, not configs; and the search runs on DeepSea development sizes, so it is charged
  to the DeepSea dev budget (150–250 runs), not to MinAtar's 240. The 10 dev cells alone consume
  110 of that, leaving 40–140 for all tuning. On the MinAtar side, 240 runs ÷ 5 pilot seeds
  minus the 12-config `K_shared` sweep = 18 draws per tuning game, exactly.

**A new gap surfaced from that audit:** the **per-candidate tuning seed count** is a value freeze
item 1 should state and does not. It is now the only genuinely discretionary number left
(3 recommended; 5 would force `n_backbone` down to 8).

**One is a process action, and it is overdue:** the external methodological review was
time-boxed, and the window has elapsed with no reviewer response on record. Under the freeze
policy the next step is mechanical — substitute reviewer, or proceed with a documented waiver.
This one is pure calendar time and does not depend on any code.

**One cannot be decided yet, by design:** the compute cap `Y` needs measured pilot wall-clock,
and the pilot runs after the freeze. That ordering is intentional, not an oversight.

---

## 5. What's next?

In order:

1. **Answer the two decisions above** (minutes). Everything downstream is blocked on them.
2. **Resolve the review window** — chase, substitute, or waive. Independent of 1, so start now.
3. **Build the sweep driver** — the selection statistic exists; the thing that *runs* a search
   and loops configurations does not. Needs decision 2's distributions.
4. **Run the backbone tuning pass** — this finally produces the well-tuned DDQN baseline the
   whole comparison rests on. First genuinely scientific result.
5. **Finish steps 7–9** — mini-searches and the diagnostics battery.
6. **Cut the final freeze tag**, apply the staged protocol fixes, mirror to OSF, then pilot.

**The critical path is decisions 1 and 2, not code.** Roughly a day of implementation work is
waiting on two numbers.
