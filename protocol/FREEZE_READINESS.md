# Final-freeze readiness — Session 1, Stage 2

> **Clock update 2026-07-30.** The Gate-1 one-week window described below has **elapsed**
> (`prereg-draft` tagged 2026-07-22; day 8 today). Per item 18 the next step is mechanical:
> silence → **nominate a substitute reviewer**; if none engages within a further week
> (~2026-08-05) → **documented waiver**. Neither has been recorded in `protocol/`, so Gate 1
> is now an *owner action that is overdue*, not a window to wait out. Gates 2 and 3 are
> unchanged. The body below is the 2026-07-23 assessment and is kept for provenance.

> **Missing-values update 2026-07-30 — CLOSED.** A pre-freeze audit found that two freeze-list
> items claimed values the draft did not actually contain (item 7 probe-set construction, item 2
> search distributions/budget), and a third defect surfaced from auditing the second (item 1 has
> no seed count for a tuning candidate). **All three are now owner-approved**, with the text and
> arithmetic in `protocol/decisions/staged_stage3_protocol_fixes.md`:
>
> | Item | Approved value |
> |---|---|
> | 7 | Probe set **exhaustive**, `\|S\| = N(N+1)/2`, no cap; **no MinAtar probe set** (the battery needs `Q*`) |
> | 2 | `n_backbone = 12`, `n_mini = 4` each of two, distributions as drafted |
> | 1 | **3 seeds per tuning candidate** (new sub-clause) |
>
> These are **staged, not applied**: `preregistration.md` is unchanged so that `prereg-draft`
> stays a valid stable reference for the Gate-1 pass, and item 18 stage 3 is the designated
> channel for incorporating them. Two consequential edits ride along at stage 3: item 1's phrase
> "8 non-overlapping derived streams" becomes **9** (`hparam_search` added to `STREAM_NAMES` on
> main, with a pinned-bytes test proving no already-derived seed shifted), and §3.4's DeepSea
> development budget should note the 230-run composition (110 cell + 120 tuning).
>
> **Consequence for this document's verdict: unchanged.** The missing values were never one of
> the three gates. Gates 1–3 below are still the binding constraints, and all three remain
> owner/process actions rather than code.

**Assessed 2026-07-23** against repo HEAD `f115f99` (all values verified live: 157 tests
pass, ruff clean, CI green on the latest push; `prereg-draft` tag confirmed on GitHub).

The final freeze is defined in `preregistration.md` as: **(2) external methodological pass on
the valued draft → (3) incorporate fixes → new tag + OSF mirror.** Stage 1 (`prereg-draft`) is
complete. This document records whether Stage 2 can be closed now.

## Verdict: NOT YET — three gates open, none of them code

| # | Gate | Status | Owner action |
|---|---|---|---|
| 1 | External methodological pass (1-week time-box) | **Window still open** | Decide: reviewer engaged, or start the silence→waiver clock |
| 2 | OSF mirror (part of the freeze definition) | **Blocked here** | Configure OSF credential + allowlist, or nominate a mirror |
| 3 | BDQN-scaling `prior × K` open item (`positioning.md` §3.4) | **Null reconfirmed on arXiv; OpenAlex re-run pending** | Confirm we close it as a documented null |

### Gate 1 — external pass window is not elapsed
`prereg-draft` was tagged **2026-07-22**. The pre-registration fixes the external pass as a
**one-week time-box** (silence → substitute reviewer; none within a further week → documented
waiver). As of 2026-07-23 we are **~1 day into a 7-day window**. No review, waiver, or
substitute record exists in `protocol/`. The freeze is *supposed* to run against a stable
reference during this window; cutting the final tag now would collapse the pass the protocol
requires. Either a reviewer's fixes must be folded in, or the silence-waiver path must be
started and documented — neither can be short-circuited from inside a work session.

### Gate 2 — OSF mirror cannot be created from here
The freeze is literally "new tag **+ OSF mirror**." `api.osf.io` is **off the network
allowlist** and **no OSF credential is configured**. The GitHub tag half is fully doable from
here; the OSF half is not. Options: (a) grant OSF network access + add an OSF token and mirror
programmatically; (b) owner creates the OSF component by hand at freeze time; (c) amend the
protocol to name a different immutable mirror (e.g. a Zenodo GitHub-release archive, which the
plan already contemplates for submission). This is a freeze-definition dependency, not
optional.

### Gate 3 — the one open pre-freeze scientific item is essentially closed
`positioning.md` §3.4 / §A.8 carry the **BDQN-scaling `prior × K` preprint** as an open item
that "must be located and characterized before the final freeze" (severity-3 threat to C-i/C-ii
on the K axis *if* it is a genuine near-twin). Re-run of the neighbor search closer to freeze
(2026-07-23):

- **arXiv** (primary venue for this class of preprint; no key needed) — six broadened queries
  in `cs.LG` (`"randomized prior" AND "ensemble size"`; `"bootstrapped DQN" AND diversity AND
  exploration`; `"posterior collapse" AND "deep reinforcement learning"`; `"ensemble size" AND
  "deep exploration"`; `"randomized prior functions" AND diversity`; `ti:"randomized prior" AND
  abs:ensemble`). **No near-twin located** — hits were Osband 2018 itself and unrelated
  operator-network / Bayesian-optimization uses of "randomized prior." Consistent with the
  Session-2 null.
- **OpenAlex** — the connector's API key did not reconnect within this session, so the
  full-text re-run is **pending**. arXiv is the load-bearing venue here, so this does not change
  the verdict, but the appendix note should record the pending OpenAlex re-run.

**Recommendation:** close §3.4 as a *documented null under the stated search protocol* (the
protocol already permits "we did not identify, under this search protocol"), with the OpenAlex
re-run completed once the connector reconnects. This gate is not a blocker on its own.

## Everything else is freeze-ready
- All 20 freeze-list items valued; Class-2 nuisance table valued (mask Ber(0.5), 1/K grad
  norm); cap X = 120 GPU-h frozen; cap Y rule + ladder frozen (scalar correctly deferred to the
  Session-3 pilot as a pre-registered amendment).
- `positioning.md` committed as single source of truth with the reproducible-search appendix.
- Method spine (DDQN, Bootstrapped-DQN, RP-BDQN, NoisyNet), DeepSea env + exact Q*, config
  loader, trainer, and the temporal-persistence diagnostic all built, tested, CI-green — this is
  pre-freeze infrastructure and does not gate the tag.

## The one-line answer
The protocol content is ready. The freeze is held open by **process, not readiness**: the
external-pass clock has ~6 days left, and the OSF half of the mirror needs credentials/allowlist
this environment doesn't have. The scientific open item (BDQN-scaling) can be closed as a
documented null now.
