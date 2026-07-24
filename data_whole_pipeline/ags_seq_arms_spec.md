# Sequential Control Arms — Implementation Spec

AGS (parallel) is already implemented. This spec covers the two **negative-control** arms only.
The emphasis throughout is on what differs; everything not listed here must be the identical code
path AGS already uses.

Three rows appear in the paper's sequential table:

| Arm | Rounds | Operator selection | Role |
|---|---|---|---|
| `AGS` | 1 | — | the method (already built) |
| `AGS+Seq` | up to B=4 | Thompson Sampling over per-operator posteriors | negative control |
| `AGS+Seq-random` | up to B=4 | uniform random from the admissible slate | control for the control |

---

## 1. What must be shared, not reimplemented

Both control arms **start from AGS's round one, byte-identical**. This is the whole point: it makes
`round-1 vs full-episode` an internal comparison rather than a comparison between two systems.

Reuse without modification:

- generator prompt and schema, J hypotheses, stochastic decoding
- both renderers (label-form, definition-form), modality-conditional
- retrieval: K=200, datatype pre-filter, label-coverage term `w_cov=1.0`
- `agree()` and the controlled vocabulary
- consolidation: sum-RRF (kappa=60) -> range-normalize -> `+ beta * consensus`, beta=0.6

**Assertion to add:** for every fact, the round-one candidate set produced by `AGS+Seq` must equal
the final candidate set produced by `AGS` exactly. If they differ, the arms are not comparable and
the round-1-vs-full column is meaningless. Fail loudly on mismatch.

---

## 2. What gets added: the sequential loop

After round one, repeat up to `B=4` total rounds:

```
FEEDBACK -> CONTROLLER -> REVISE -> RENDER -> RETRIEVE -> (re-consolidate)
```

### 2.1 Feedback stage

Take the top `M=10` candidates from the current round's ranking, selected as cluster
representatives over the retrieved list under the symbolic-dimension profile (not the raw top 10 —
those are lexical near-duplicates and carry little contrast).

Run `agree()` per candidate per dimension and aggregate into:

```
D_plus     dimensions supported by the neighborhood
D_minus    dimensions contradicted
D_question dimensions unresolved / undeterminable
g          structural-mismatch indicator
n          neighborhood novelty vs previous rounds (Jaccard-based)
```

**Symbolic verdicts only.** Do not add an LLM verification call — it measured at base rate and the
paper reports it as disabled. If an LLM layer already exists behind a flag, keep it off.

The feedback stage never sees the gold concept.

### 2.2 Controller: build the admissible slate

Propose at most `L=6` atomic directives, one per operator. A directive is
`(mode, operator, target_dim, patch, preserve_set)`.

Admissibility is determined by feedback, identically in both arms:

```
REFINE          target one dimension in D_minus
BRANCH          test an alternative for one dimension in D_question,
                preserve everything in D_plus
CHANGESTRATEGY  replace the interpretation operator when g fires
PERTURB         change no dimension; re-render under stochastic decoding
```

Each directive modifies exactly one semantic factor. `preserve_set` must be enforced by the
revision step, not merely suggested.

### 2.3 The only difference between the two arms

```
AGS+Seq:         sample theta_o ~ N(mu_o, nu^2 Sigma_o) for each operator in the
                 slate; select argmax over psi^T theta_o
AGS+Seq-random:  select uniformly at random from the slate
```

**Both arms still compute and update the posteriors.** The random arm just does not consult them
for selection. This is what makes the comparison interpretable: posteriors moving only matters if
consulting them improves behavior.

### 2.4 Revise and re-retrieve

```
h_{t+1} = revise(x, h_t, directive)   # applies only the authorized patch
render -> retrieve -> add to accumulated pool U
re-consolidate over U using AGS's consolidation function
```

---

## 3. Decision to make before launching: the novelty gate

The 250-instance diagnostic ran with a pre-retrieval token-Jaccard gate that rejected revisions
too lexically similar to a previous query. It terminated 90% of episodes at a realized 2.34 of 4
rounds, because an atomic single-dimension edit barely changes the rendered query.

**Recommendation: run the test arms with the gate OFF.**

Rationale: the gate was intended to save budget, but the expensive part (the revision call) is
already spent when it fires, and BM25 retrieval is nearly free. Leaving it on means the negative
result is reported on a system that never got to use its own budget — a reviewer will say the
sequential arm was handicapped. With the gate off, realized rounds should reach ~4 and the arm
gets its strongest form.

Consequence to handle in the paper: the appendix diagnostic (gate on, 2.34 rounds) and the test
table (gate off, ~4 rounds) then describe different configurations. That is fine as long as both
are labeled. Report realized rounds per fact in the efficiency table either way.

If you keep the gate on for continuity with the appendix, cap rejections at one per round and log
the rejection rate.

---

## 4. Delayed learning (both arms)

Only after an episode terminates and the gold concept is revealed.

**Utility on the consolidated ranking**, not on a single query:

```
U_t = union of candidate lists through round t
u_t = 1 / log2(1 + rank_t(gold))    if gold in U_t, else 0
      # rank_t under AGS's consolidation score computed over U_t
delta_y_t = u_{t+1} - u_t
```

Log-discounted and over the untruncated union — this is deliberate, so that a round which first
brings gold into reach earns credit even if it does not yet place it in the top-K.

**Counterfactual replay:** freeze the slate and the runner-up directive at decision time. After
gold is revealed, replay the revision for the runner-up under deterministic decoding, retrieve,
compute `u_tilde` over `U_t + C_tilde`, and set `delta = u_{t+1} - u_tilde`. Stored credit is
`r_t = alpha * delta_y_t + (1 - alpha) * delta`.

**Per-operator posterior**, symbolic context `psi` only, with forgetting factor `zeta`:

```
A_o = lambda*I + sum over records of zeta^(delta_i) psi psi^T
b_o = sum over records of zeta^(delta_i) psi r
mu_o = A_o^-1 b_o ;  Sigma_o = sigma^2 A_o^-1
```

Reduce the `psi` feature set before running: the diagnostic reached a condition number of 4,843,
which means the features are badly collinear and `mu` is unstable. Compute the pairwise
correlation matrix on the existing logs, drop or merge blocks with |r| > 0.9, and report the
retained dimension and the new condition-number trajectory.

**Memory:** store symbolic feature blocks, operator, and credit. Never store gold identities.
Exclude records from the same source context as the current instance — without this, apparent
online gains are confounded by the 21 facts per table in this benchmark.

---

## 5. Required outputs

Same instance order for both arms. Text facts must be included this time; the earlier diagnostic
was tabular-only, which left a visible hole.

```
per_fact.jsonl
  fact_id, context_id, modality, arm
  round1_candidates, final_candidates      # both, for the round-1 vs full column
  round1_rank_gold, final_rank_gold
  realized_rounds, stop_reason

rounds.jsonl
  fact_id, arm, round_idx
  psi (named dict), slate, selected_operator, selected_mode,
  runner_up_operator
  D_plus / D_minus / D_question counts, split by exact vs lexical branch
  gold_in_union_before, gold_in_union_after
  rank_before, rank_after                   # consolidated, over the union
  delta_y, delta_replay, reward
  candidate_list                            # per round, so consolidation
                                            # variants remain evaluable offline

arm_summary.csv
  arm, R@10, R@50, R@200, MRR, coverage, top1_accuracy
  round1_R@50, full_R@50, difference        # the paper's key column
  realized_rounds_mean, AULC
```

Metrics paired per fact, bootstrap CIs at the source-context level, 2,000 iterations, table and
text reported separately.

---

## 6. What the run is for

Not to show the sequential arms are good. To put on the test split three numbers the paper
currently supports only with a 250-instance tabular diagnostic:

1. `full_episode - round1` at R@50 — expected to be at or below zero
2. `AGS+Seq` vs `AGS+Seq-random` on AULC — expected: random is not worse
3. both against `AGS` — expected: neither exceeds it

If any of these comes out the other way, the paper's Section 5 changes rather than the run being
repeated. Report what the run produces.
