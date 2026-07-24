# A1 — Claude Science Execution Plan

**"When Do Randomized Value Estimates Buy Exploration? Separating the Estimator from the Use Rule under Low Interaction Budgets."**

**Robert Meliksetyan · July 2026 · Execution plan v4.3 (built on A1 requirements spec v6.3 and pivot plan v4)**

**Changelog v1 → v4.1:** as recorded (mechanism restructure + gates; development/confirmatory sequencing; factorial + C0; QR-DQN exploratory; shared config + C13; estimand hierarchy; Variant B; failure policy; two-stage freeze; amendment policy; parameter classes; K_shared rule; deterministic conditionals).

**Changelog v4.1 → v4.2:** stale Session-1 line replaced; spec references synced; pre-launch frozen horizon wording; rung-2/C-K conditional; Brief 1 additions; Brief 2 rebuilt on the contribution map; risk register updated.

**Changelog v4.2 → v4.3 (review #7 sync):**
- **"10-cell factorial" → "10-cell structured partial factorial"** everywhere (full `use_rule × prior` core at K=10 + prior × K axis for episodic; K × use_rule and three-way interactions not estimable — limitation stated).
- **Cell-specific RNG derivation** added to conventions and Brief 0/3: every stream (init, environment mapping, replay, action noise) derived as `hash(master_seed, cell_id, stream_name, seed_index)`; no stream reused across cells; the independent stratified bootstrap is justified by construction. The v6.2 "same integer seed creates no pairing" claim is retired as incorrect.
- **Development seed counts pinned:** exactly 10 dev seeds for the canonical episodic cell and the ε-greedy reference (they are the frozen inputs of both the t₀ rule and the size-selection rule); 5 dev seeds for the remaining cells.
- **Mini-search objectives pinned** (Session 5): `prior_scale` by IQM of `(episodic, on, 10)` on dev sizes; `eps_schedule` by IQM of `(mean_eps, off, 10)` on dev sizes; ties → lower parameter value.
- **Brief 2 extended:** EVE, Priors Matter (2025), and the July-2026 distributional-risk audit added (existence of the audit verified 2026-07-15; content characterizations to be read against the papers); BDQN-scaling promoted to the top neighbor tier; narrowed null-result searches; "we did not identify" language; the corrected novelty audit + **reproducible-search appendix** fold into `positioning.md` as the single source of truth (the standalone novelty deep dive is retired).
- **Fairness/firstness language:** "we apply the standard established by Ceron and Taïga" — the word "first" banned; reviewer agent flags "full factorial," "first," and "no paper exists."
- **Grant note converted to final-call** (deadline is today, July 15) with an archive instruction.

---

## 0. What this document is — the *what* lives in `a1-requirements-and-alternatives-v6.3.md`; this is the *how* inside Claude Science (native reproducibility; burst compute via Modal/Slurm for tuning; reviewer agent for claim traceability). The judgment layer guards, in order: **identification** (contrast purity incl. configs — C11 + C13), **confirmatory integrity** (sequencing + §3.6 failure policy), **the estimand hierarchy** (no post-hoc promotion), **statistical independence** (the cell-specific RNG derivation), and the **frozen-rule/measured-input discipline**.

---

## 1. Objective and success bars

Objective: the v6.3 two-part flagship — Part A: pre-registered structured-partial-factorial replication-and-extension on DeepSea; Part B: external performance evaluation on MinAtar — plus public repo and RLC-2027 submission (verify CFP; Q1 2027 is a working assumption).

- **EWRL bar (Oct 5–7, Lille — dates confirmed):** primary = confirmatory mechanism figure (post-Gate-A block); fallback = pilot-labeled development mechanism figure; second fallback = pilot solve-vs-depth. The one-sentence pitch: *"We're re-running the Bootstrapped-DQN mechanism ablations as a pre-registered, exact-ground-truth structured design — here's the primary contrast."*
- **RLC bar (Q1 2027):** the v1.0 gate from spec v6.3 §6.

Nothing is "done" until its box is checked *and* the reviewer agent raises no untraced claims.

---

## 2. Feature map — unchanged (code execution, burst compute, artifact tracing, figure iteration, reviewer agent, literature analysis, manuscript drafting = core/high; life-science databases = ignore).

## 3. Session 0 — Bootstrap: env + lockfile; repo skeleton (`protocol/` committed first); CI smoke; burst-target check; **throwaway infrastructure benchmark** (informs caps X and Y only; logged as `infra_benchmark.md`; not a scientific result); **the cell-specific RNG derivation utility implemented and unit-tested here** (hash-derived, non-overlapping streams per (cell, stream, seed-index)). Attach `a1-requirements-and-alternatives-v6.3.md` and the pivot plan as project context.

## 4. Reproducibility contract — portable repo is the deliverable; Claude Science tracing is the in-session net; disagreement is a hard-stop bug; the pre-registration is timestamped by the two-stage tags (`prereg-draft`, final) + OSF mirror.

---

## 5. Compute plan

Two lanes: **burst** (tuning sweeps) and **local** (reported runs). **Run-budget table (spec v6.3):**

| Block | Approx. runs | Lane |
|---|---|---|
| Backbone tuning pass (DDQN nuisance, dev sizes) + two factor-specific mini-searches | ~60–100, cheap | Local |
| MinAtar tuning pilot tier (canonical four; K ∈ {5,10,20} searched by both ensemble methods) | ~240 @500k | Burst |
| MinAtar tuning final tier (**method-specific formula-gated**) | ~800 @500k | Burst |
| MinAtar held-out evaluation (**unconditionally to the pre-launch frozen horizon: 1M default / 500k if rung 1 pre-launch; checkpointed**) | ~120 | Local |
| DeepSea dev pilots (**10 seeds × 2 rule-input runs; 5 seeds × remaining cells**) | ~150–250 | Local |
| DeepSea confirmatory sweep (10-cell structured partial factorial × 5 selected sizes × 20 seeds + baseline/NoisyNet ride-along) | ~1,100 | Local, cheap |
| QR-DQN exploratory follow-up (post-Gate-B) | ~300 | Burst + local |

**Final-tier trigger:** final tier runs iff Σ over the four methods of (final-tier run count × that method's **own** median pilot-tier per-run wall-clock) ≤ X GPU-hours; X frozen in Session 1; inputs measured in Session 7's pilot tier; formula applied in the open, logged in `protocol/trigger_input.md`.

**Descope ladder (trigger per spec freeze item 17):** applied only by the frozen compute-cap rule (projected total vs. cap Y after pilot wall-clock measurement), **before Gate A and before any confirmatory outcome**; rung 1 only before held-out launch. Order: (1) pre-launch horizon → 500k (RQ3 descoped) → (2) K-axis → {10, 20} — *C-K exits the confirmatory sequence per spec §1.1* → (3) final-tier configs down (identical across methods) → (4) held-out seeds 10 → 5 with sensitivity note → (5) drop MinAtar E-cell spot-check → (6) drop QR-DQN follow-up. **Never cut:** the **six-cell `use_rule × prior` core at K = 10**, dual axes, the EWRL figure, the protocol, DeepSea confirmatory seeds.

---

## 6. Non-negotiables every session inherits

- Equal search budget across the **canonical four** — **the matched-search-budget standard established by Ceron and Taïga, applied to randomized-value exploration methods** (never "first," never original-fairness); sensitivity curves reported. QR-DQN follow-up mirrors it for comparability only.
- Both contamination surfaces (MinAtar tuning/eval split; DeepSea development/confirmatory split) — C8.
- **The 20-item freeze list** (spec v6.3 §3.1), notably: the **cell-specific RNG derivation scheme**, the **three-class parameter rule** (backbone tuned on DDQN; ensemble-shared fixed via the explicit **parameter table**; factor-specific `prior_scale`/`eps_schedule` with **pinned selection objectives and tie-breakers**), the **primary estimand hierarchy with literal gatekeeping and SESOI Δ=0.10**, the **method-specific trigger + X**, the **descope trigger (Y) with the rung-2/C-K conditional**, the **K_shared joint rule**, the **exact size-rule thresholds and their 10-seed input populations**, the **t₀ rule with its pinned input population** (the canonical episodic cell's 10 dev seeds), the **failure policy**, **battery formulas + micro-conventions + undefined-value policy**, the **deterministic cloning conditional**, the **canonical four**, QR-DQN's exploratory status.
- **Two-stage freeze + amendment policy:** draft tag → external pass (silence → substitute reviewer → documented waiver) → final tag + OSF → runs; post-freeze/pre-confirmatory changes only as versioned amendments; post-confirmatory only under the confirmatory-integrity rule.
- Frozen-rule + measured-input pattern; no deferred choices.
- Both evaluation axes; direct exploration outcomes; probe sets; language rules ("replication and extension"; "external performance evaluation"; "structured partial factorial" — **"full factorial" banned**; contrast names for mechanism claims; "we did not identify," never "no paper exists" or "first"; "associated/anticipates," never causal "predicts").
- **Primary statistics: probability of improvement, raw per-game differences, performance profiles. No ratios.** Three-per-game MinAtar reporting; no general-ranking claim.
- **Confirmatory-integrity rule with §3.6 as its operational arm.**
- **C0:** no correctness gate conditions on studied-method rankings.

**Session 1 deliverable (two-stage):** the valued 20-item pre-registration under a `prereg-draft` tag + the v6.3 advisor one-pager (built per spec §9's content rule, safe novelty sentence verbatim), sent as touch #2 (Gate A track) and to the external reviewer (1-week box; substitute/waiver rule); then the **final pre-registration tag + OSF mirror** after fixes. Scientific runs begin only after the final freeze; the draft-vs-final diff is committed.

---

## 7. Session breakdown

### Session 1 — Value and freeze the protocol, two-stage (Robert-led)
- **Stage 1 — valued draft:** fill all 20 freeze-list items → `prereg-draft` tag; advisor package out (Gate A track); external reviewer engaged in parallel.
- **Stage 2 — final freeze:** incorporate fixes → final tag + OSF. Sessions 0 and 2 may proceed during the external-pass window; Session 3 onward requires the final freeze.
- **Your judgment:** every numeric value; the hierarchy and SESOI; caps X and Y; the size-set triple and thresholds; the parameter table; the mini-search objectives; the failure and amendment policies.
- **Gate:** final tag + OSF exist before any Session-3 run; draft-vs-final diff committed; advisor and external-pass records logged.

### Session 2 — Positioning / scoop-check (delegate)
> Positioning read and scoop-check for the v6.3 contribution map; `positioning.md` is the **single source of truth** and absorbs the corrected novelty audit. Verify against the papers themselves: Osband 2016 (ensemble-policy and per-step "Thompson DQN" sketches — exact figures/sections for the replication claim, plus mask-p and normalization settings for the parameter table), Osband 2018/2019 (RVF — the paper we extend), **the BDQN-scaling preprint (top neighbor tier: prior × K on DeepSea, posterior collapse, diversity — verify review status)**, **EVE / Exploration via Epistemic Value Estimation (continuous epistemic uncertainty on DeepSea; acting-rule and bootstrapping ablations — neither "all prior work is whole-algorithm" nor "all DeepSea work is pass/fail" may be claimed)**, **Priors Matter 2025 (prior/likelihood misspecification in Bayesian DQN; MinAtar + DeepSea — frames C-PRIOR as complementary)**, **Auditing the Risk Claims of Distributional RL (July 2026; existence verified 2026-07-15 — read and characterize; it confirms QR-DQN's exploratory weight and scopes C-ii to randomized-value estimates on DeepSea)**, Janz 2019 (analytic precursor), MULEX + Clements 2019 (different-axis disentanglements), Plappert 2018, HyperAgent + bsuite (pass/fail usage; DeepSea introduced in the randomized-prior line, standardized in bsuite), Ceron 2021 + Taïga 2021 (inherited standard; draft the Taïga pre-emption paragraph), UA-DQN, β-DQN, Neural Testbed. Re-run the **narrowed** null searches ("the exact canonical set under one matched MinAtar protocol"; "bootstrapped + NoisyNet joint") with "we did not identify under this search protocol" phrasing. Produce the **reproducible-search appendix**: search date, databases, exact queries and filters, anchor papers, citation-walk sources, retrieval counts before/after deduplication, screening rules, full-text list, null queries, limitations. Ban "full factorial," "first," "no one has," and original-fairness language.

### Session 3 — Pipeline spine (post-final-freeze)
DDQN on Breakout, 1 seed → determinism discipline (**cell-specific derived streams in force from the first run**) → `make figures` curve → 5-seed baseline. Per-method wall-clock recorded as each method lands.
**Gate:** C1–C3.

### Session 4 — Switchboard + backbone tuning + diagnostic
- **Objective:** the structured-partial-factorial switchboard (one file; cells by `use_rule/prior/K` config) **plus the backbone-tuning pass under the three-class rule**: tune backbone nuisance only on the DDQN backbone on development sizes; **fix ensemble-shared nuisance via the parameter table** (per-value citations from Osband 2016 or documented design choices); freeze both; all cells inherit them. Disagreement logging on frozen probe sets.
- **Gate:** C4; C11 (per-contrast code-path purity); **C13 first run** (re-audited in Session 5 once factor-specific params exist).

### Session 5 — DeepSea development integration + factor-specific mini-searches + NoisyNet
- **Objective:** DeepSea on development sizes N = 10, 20 only; pilot solve-vs-depth (pilot-labeled; second-fallback EWRL figure); the two mini-searches under their **pinned objectives**: `prior_scale` by IQM of `(episodic, on, 10)` on dev sizes (shared by all prior=on cells); `eps_schedule` by IQM of `(mean_eps, off, 10)` on dev sizes (shared by its cells); ties → lower parameter value. Then NoisyNet.
- **Gate:** C5 (environment/implementation only: dynamics vs. spec; exact Q\* on a hand-checkable size; oracle solves both dev sizes; random/ε-greedy-reference degradation N=10→20; reference-behavior match); C0 restated; C6; C13 re-audit. Touch no confirmatory size.

### Session 6 — Battery + feasibility spike
Battery per the frozen formulas of spec §3.3 (M = K heads; M = 30 NoisyNet draws at measurement; 80% containment via `numpy.quantile(..., method="linear")`; same-actions g\* with the lowest-index tie rule; rank-biserial formula and sign; per-checkpoint OLS decay; undefined-value policy). Q\* validated on a hand-checkable size first; frozen probe sets; every diagnostic at every checkpoint incl. t₀. **MinAtar cloning spike under the deterministic conditional** (100 clone/restore tests → full / episode-start-only / drop; raw outputs committed). Touch no confirmatory size.
**Gate:** C7.

### Session 6b — Mechanism pilot (development sizes; pre-Gate-A safe)
> Confirmatory-integrity rule in force; development sizes and tuning games only. Pilot the 10-cell structured partial factorial under the shared config: **exactly 10 development seeds for the canonical episodic cell `(episodic, off, 10)` and the ε-greedy reference** (these runs are the frozen inputs of both the t₀ rule and the size-selection rule), **5 seeds for the remaining cells**, logging the battery at every checkpoint. Measure the t₀ input (fraction of the canonical cell's 10 seeds with a first success by the 10% checkpoint; frozen >2/10 → 5% fallback). Apply the size-selection rule in the open (≤2/10 down; ≥9/10 up; down-shift precedence) and record its output set. Compute and commit the **prospective MDE report** from development-pilot variance. Produce the pilot mechanism figure (episodic vs per_step vs mean_eps; priors and K as panels), pilot-labeled — the primary fallback for EWRL. Every claim cites a named contrast. Run no confirmatory size.
**Gate:** C11 + C13 per contrast; all outputs pilot-labeled.

### Session 7 — MinAtar tuning → GATE A → confirmatory block → M1
- **7-pre:** two-tier tuning of the canonical four on the tuning games (burst); **K_shared rule applied** (K ∈ {5,10,20} searched by both ensemble methods; K_shared = argmax of their mean best-config IQM; both inherit it); per-method pilot-tier wall-clock medians recorded; **the method-specific trigger formula and the descope compute-cap trigger applied in the open** (rungs in frozen order until projected ≤ Y — before Gate A, before any confirmatory outcome; rung 1 only applicable now, pre-launch); configs frozen by IQM. **STOP: Gate A.**
- **7a (post-Gate-A):** DeepSea confirmatory sweep, exactly once — the 10-cell structured partial factorial × the rule-selected sizes × 20 seeds under the shared config, baseline + NoisyNet riding along; battery at every checkpoint incl. t₀; **the §1.1 primary estimand computed exactly as frozen before anything else**; the confirmatory mechanism figure (primary EWRL figure).
- **7b (post-Gate-A):** MinAtar held-out evaluation — **runs proceed unconditionally to the pre-launch frozen horizon (1M default; 500k if rung 1 triggered pre-launch); once launched, no selective continuation or early stopping**; checkpointed; both axes (C12); direct exploration outcomes; capacity/compute table; probability of improvement + raw differences + profiles; three per-game results primary.
- **Gate:** **M1/v0.1 release**; C8 (both splits); C9 (equal-tuning audit); C12; C13 committed; §3.6 failure log clean or documented; confirmatory-integrity rule in force downstream.

> ### ▶ GATE A — advisor protocol review (before both confirmatory blocks)
> Frozen protocol + freeze list + pilot evidence from development sizes and tuning games only; the rule-selected size set reviewed as an *application of a frozen rule*. Time-box: 3 weeks → documented deviation + substitute review. **Any design change recommended here is enacted through the amendment policy** — versioned amendment, new tag + OSF timestamp, reasons + changed items + motivating dev outcomes disclosed; confirmatory runs only after the amendment freeze.

> ### ▶ GATE B — advisor results review (before any expansion)
> Interpretation and exploratory extensions only; the confirmatory design is closed. Sees budget-axis data by design (harmless — no RQ3 design decision remains open). No new methods enter any confirmatory analysis — the QR-DQN follow-up is exploratory by pre-registration. Same time-box. Both gates are letter-production activity.

### Session 8 — QR-DQN exploratory follow-up (post-Gate-B)
> Implement and run QR-DQN under the mirrored search budget; test the pre-registered epistemic-vs-distributional hypothesis under the neutrality rule; label every artifact **exploratory**; touch no confirmatory table or figure. Cite the distributional-risk audit paper as kindred related work for this block.

### Session 9 — RQ3 secondary analysis (analysis-only)
> No new runs. Aggregate the existing held-out checkpoints per the pre-specified secondary plan; state the tune-once-at-500k confound; **if the rung-1 horizon applied, report RQ3 as descoped**. Run the local single-GPU honesty re-run and log total GPU-days here.

### Session 10 — Aggregation + RQ2-L + paper figures
> Confirmatory-integrity rule; no design changes. Freeze paper statistics per spec v6.3: probability of improvement, raw per-game differences with stratified bootstrap CIs, performance profiles (no ratios); both axes; per-budget and per-env tables; three-per-game boundary. **The §1.1 hierarchy reported exactly as frozen** (incl. the rung-2/C-K substitution if triggered). **RQ2-L per its full specification** (marginal alignment at t₀; within-stratum standardization; comparable-pair concordance; pair-count-weighted combination; 10,000 within-stratum permutations; fixed strata; <10-pair exclusion). Cross-method scatter as descriptive appendix. Tuning sensitivity curves published; QR-DQN plotted as the labeled exploratory contrast outside.
**Gate:** C10; no unlabeled pilot data; no post-hoc promotion; `make figures` end-to-end.

### Session 11 — Manuscript + reviewer pass
> The two-part paper per spec v6.3 §5: replication-and-extension abstract with verified Osband citations; the Taïga pre-emption paragraph; the partial-factorial interaction limitation stated; Part B titled "external performance evaluation" with the three-game boundary; RQ3; limitations; QR-DQN in a marked exploratory section. Related work from `positioning.md` (which carries the reproducible-search appendix) and the A2 corpus (structured bibliometric survey). Reviewer-agent trace pass on every quantitative claim.
**Gate:** **v1.0** (spec v6.3 §6); zero untraced claims; limitations advisor-signed.

### Session 12 — Ceiling (per-item go/no-go behind Gate B): E5 + spread-temperature · ensemble_mean_greedy cell · per-budget re-tune spot check · MiniGrid/Classic Control · ensemble-Thompson/PSRL · bonus anchor.

---

## 8. Correctness & validation gates

**C0** — no correctness gate conditions on the ranking of studied methods/cells. **C1–C3** — determinism (incl. the derived-stream scheme); logging integrity; baseline sanity vs. published MinAtar scores. **C4** — ensemble diversity, masking. **C5** — environment/implementation only. **C6** — NoisyNet correctness. **C7** — ground truth + battery per frozen formulas. **C8** — no contamination on either split. **C9** — equal-tuning audit. **C10** — aggregation discipline; hierarchy honored. **C11** — per-contrast code-path purity. **C12** — frozen-policy extraction. **C13** — configuration-identity audit (committed with the figures).

`VALIDATION.md` ticks per session; any failure is a hard stop; §3.6 governs failures in confirmatory blocks.

---

## 9. Reviewer agent — standing checklist

- Every mechanism claim cites a **named contrast** and its C13 audit entry; causal language only for C-USE/C-COHERENCE; C-PRIOR/C-K say "estimator intervention."
- Flag: promotion of any non-primary contrast/outcome/diagnostic to confirmatory language; QR-DQN outside exploratory labeling; **"full factorial," "first," "no paper exists,"** "transfer," original-fairness phrasing; ratio metrics; a fifth method in a confirmatory MinAtar aggregate; selective continuation or seed-count changes post-results; "held-out" applied to DeepSea; ranking-conditioned checks (C0); pre-Gate-A artifacts touching confirmatory sizes; unlabeled pilot evidence in confirmatory sections; **any reuse of an RNG stream across cells**.

---

## 10. Milestones as checklists

**M1 / v0.1 (end of Session 7):** canonical four on MinAtar (pilots) · switchboard + shared config (C11+C13 green) · dev mechanism pilot (labeled; 10/5-seed structure) + MDE report · size rule applied · confirmatory sweep once + §1.1 primary + mechanism figure · MinAtar held-out to the pre-launch frozen horizon, both axes, three-per-game reporting · EWRL figure · README · repo public.

**Gate A / Gate B:** as above.

**v1.0 (Session 11):** hierarchy reported as frozen · structured partial factorial complete — incl. C-K, or with the rung-2 descriptive substitution documented · RQ2-L per spec · both axes at all checkpoints · sensitivity curves + capacity/compute table · C13 outputs committed · RQ3 secondary analysis (or descope documented) · QR-DQN exploratory section (or descope documented) · `make figures` end-to-end · compute reported · zero untraced claims · limitations advisor-signed.

---

## 11. Critical-path timeline (best-case; the frozen descope trigger absorbs slippage)

| Window | Sessions | Output |
|---|---|---|
| **Weeks 1–2** | 0, 1, 2 | Env + benchmark + RNG-derivation utility; **valued 20-item draft (`prereg-draft` tag)** → external pass (1-week box; substitute/waiver) alongside Session 2 → **final tag + OSF**; advisor package out with the draft (Gate A initiated); `positioning.md` with verified citations, new neighbors, narrowed nulls, reproducible-search appendix. Session 3 starts only after the final freeze |
| **Weeks 3–5** | 3, 4, 5 | Spine; switchboard + backbone-tuned shared config + parameter table (C11+C13); dev DeepSea; pinned mini-searches; NoisyNet; pilot solve-vs-depth (labeled) |
| **Weeks 6–7** | 6, 6b | Battery (frozen formulas) + cloning spike; **mechanism pilot** (10/5-seed structure) → pilot figure; t₀ + size-rule inputs + MDE report |
| **Weeks 8–9** | 7 | Tuning (K_shared; trigger + descope formulas in the open) → **GATE A** (amendment channel if needed) → confirmatory sweep + **primary estimand** + mechanism figure; MinAtar held-out **to the pre-launch frozen horizon** → **v0.1; repo public** |
| **Weeks 10–11** | Gate B → EWRL prep | Results review (letter activity); figures + repo polished |
| **Oct 5–7** | — | **EWRL, Lille** — the replication-and-extension pitch + mechanism figure + repo |
| **Oct–Nov** | 8, 9 | QR-DQN exploratory follow-up; RQ3 secondary analysis; honesty re-run. *Month-4 income gate early Nov.* |
| **Nov–Dec** | 10, 11 | Aggregation + RQ2-L; manuscript + reviewer pass; **v1.0**. *EDIC Dec 1 only if a professor is warm.* |
| **Q1 2027** | submission | **RLC 2027** (verify CFP when posted) |
| **As capacity allows** | 12 | Ceiling |

If Gate A stalls to its time-box in week 9: deviation documented, confirmatory block proceeds — the DeepSea sweep is ~a day of local runs, so EWRL does not move.

---

## 12. Anti-scope-creep guardrails — per-session artifact allow-lists; closed cut list; gates are walls; ceiling opt-in; new cells and follow-ups are ceiling items (E5 and ensemble_mean_greedy the only pre-approved candidates, both exploratory); **completing the 18-cell full factorial is explicitly on the cut list** — the missing cells are a documented limitation, not a to-do.

## 13. Risk register

| Risk | Mitigation |
|---|---|
| Silently wrong RL runs | C0–C13 every session; reviewer agent; Gate A before confirmatory blocks |
| Hallucinated numbers | Trace-check after every results session |
| **Configs, not code, break contrast purity** | Three-class rule + parameter table; C13 committed with every contrast figure |
| **Hidden cross-cell dependence via shared RNG streams** | Cell-specific derived streams (hash-based, non-overlapping); unit-tested in Session 0; agent flags stream reuse |
| **Post-hoc estimand shopping** | §1.1 frozen hierarchy; literal gatekeeping + SESOI; agent flags promotions |
| **Freeze/review sequence inverted or Gate A changes off the record** | Two-stage freeze; amendment policy with versioned tags and disclosure |
| **Descope applied ad hoc / rung 2 breaks C-K silently** | Frozen compute-cap trigger at 7-pre; rung 1 pre-launch only; rung-2/C-K conditional frozen |
| **Novelty destroyed or overclaimed** | Corrected contribution map (structured partial factorial; "we did not identify"; no "first"); EVE/BDQN-scaling/Priors-Matter/risk-audit engaged in `positioning.md` with verify-at-Session-2 discipline; narrowed nulls re-run at freeze and submission; agent flags banned phrases |
| **QR-DQN re-contaminates the holdout** | Pre-registered exploratory status; agent flags any confirmatory appearance |
| **Selective continuation on the budget axis** | Pre-launch frozen horizon; unconditional continuation once launched; §3.6 only exception |
| **Confirmatory failures handled ad hoc** | §3.6 operational policy frozen |
| **Trigger underestimates compute** | Method-specific pilot-tier inputs; conservative X |
| **Ceiling/floor confirmatory sizes** | Frozen three-way size rule with exact thresholds and pinned 10-seed input populations |
| Contamination · hypothesis-as-gate · deferred choices · scale/best-case · fairness critique · burst-lane honesty · agent over-trust · advisor latency · bsuite rot · cloning infeasibility | As prior versions |

## 14. Session hygiene — one project, many sessions; spec + protocol attached at project level; persist to files, not chat; restate the session's non-negotiables at start; from Session 6b onward every brief opens with the confirmatory-integrity rule + the dev/confirmatory boundary; from Session 7 onward also the §3.6 failure policy.

## Appendix A — Environment spec: Python 3.11; `torch` (CUDA local build); `gymnasium`; `minatar` (version + per-game stochasticity settings pinned and named); `bsuite` (fallback: single-file DeepSea validated against reference behavior); `minigrid` (ceiling only); `rliable`; `numpy pandas scipy matplotlib seaborn`; `pyyaml` + `tyro`; `wandb` (optional; CSV is truth); dev: `pytest`, `ruff`. Emit lockfile.

## Appendix B — Repo skeleton: `src/` (`dqn.py`; `bootstrapped_dqn.py` — STRUCTURED PARTIAL FACTORIAL SWITCHBOARD, `use_rule × prior × K` by config; `noisynet_dqn.py`; `qr_dqn.py` exploratory; `utils/` incl. the RNG-derivation utility; `diagnostics/`), `analysis/` (`rq2l/`; `hierarchy/`), `audits/c13/`, `protocol/`, `related_work/` (`positioning.md` incl. the reproducible-search appendix). CI asserts per-contrast purity (C11), the factorial config schema, the derived-stream scheme, and the smoke test.

## Appendix C — Conventions: config schema with `use_rule`, `prior`, `K`, `arm` alias fields; **every random stream derived as `hash(master_seed, cell_id, stream_name, seed_index)` — no stream reused across cells; seed labels are bookkeeping only**; `role: confirmatory|development|exploratory` on every row (QR-DQN exploratory by construction); `size_class: development|confirmatory`; `axis: online|frozen_policy`; checkpoint field incl. the t₀ flag; every run's resolved config serialized and committed (C13 input); pilot labels; CSV contract; reproducibility hash.

## Appendix D — Session briefs: each session's quoted objective block in §7 is its brief. Brief 1 additionally enumerates, verbatim from spec v6.3: the parameter table; the RNG-derivation scheme; the pinned mini-search objectives and tie-breakers; the t₀ and size-rule input populations (10 seeds, canonical cell + ε-greedy reference); the full C-K specification; the rung-2/C-K conditional; the battery micro-conventions; the substitute/waiver rule; the amendment policy; the horizon formulation; the one-pager content rule with the safe novelty sentence.

---

## Timely note — Claude Science / AI-for-Science grant: **the deadline is today, July 15 — final call.** If the application takes under half a day, submit now with the protocol one-pager; otherwise drop without regret. **Archive this note after today** (move to the changelog) so the plan does not ship with a stale deadline. Not a plan dependency.

---

*One-line version: one primary contrast, one shared config, one derived random stream per cell, one pass over the confirmatory data — and a novelty story that says "we did not identify," never "no one has."*
