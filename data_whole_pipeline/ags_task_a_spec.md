# Task A — Specialized Generators vs. Stochastic Sampling

Tests whether replacing $J$ stochastic samples of one prompt with $J$ **functionally specialized**
generators improves the ensemble. This is a configuration decision, so it runs on the **development
sample**, not test.

Current frozen AGS uses `J=2` stochastic samples. That is the baseline to beat.

**Cost:** one generation pass (~3.3k calls). Everything after is offline on the logged retrievals.

---

## 1. What is shared, not reimplemented

Reuse the existing AGS code path unchanged for everything except the generator prompt:

- output schema over the six dimensions, `null` for unresolved, structured decoding
- both renderers (label-form, definition-form), modality-conditional: dual for table, definition
  for text
- retrieval: `K=200`, datatype pre-filter, label-coverage term `w_cov=1.0`
- `agree()` and the controlled vocabulary / normalization map
- consolidation: sum-RRF (`kappa=60`) → range-normalize → `+ beta * consensus`

**Sample:** the frozen 661-fact / 70-context development sample (`sample_facts.jsonl`), the same
file used by component validation. Same fact ids for every arm.

---

## 2. The five generators

Five prompts, same schema, **deterministic decoding** — specialization replaces temperature as the
source of variation. Each is asked for its *best* reading under one structural prior, not for a
reading different from the others.

```
G0  general       Anchor on the nearest explicit descriptor of the target.
                  Commit only to attributes supported by the local context.
                  (This is the existing AGS prompt, unchanged.)

G1  row_column    Compose the row category with the semantic role of the
                  target column. Distinguish labels, metrics, events, and
                  status fields.

G2  temporal      Establish the reporting reference point. Convert calendar
                  years, age buckets, and ordered columns into relative
                  periods. Commit period type.

G3  aggregation   Resolve totals, subtotals, gross vs net, "less" rows,
                  bridges, and residual categories. Commit the qualifier.

G4  dimensional   Separate the core concept from contextual dimensions:
                  segment, region, plan, class, subsidiary.
```

Each must still emit `null` for dimensions its own reading does not support. A temporal specialist
that invents a `FAMILY` value is worse than one that abstains.

Retrieval: 2 queries per generator on table facts, 1 on text. Log every ranked list.

---

## 3. Arms

All offline from the logged retrievals except the generation pass itself.

```
S0  J=2 stochastic samples of G0            [frozen AGS baseline]
S1  G0 + G1 + G2                             3 generators
S2  G0 + G1 + G2 + G3 + G4                   5 generators
S3  S2, restricted to generators whose hypothesis passes the symbolic
    compatibility filter (datatype / period type / balance)
S4  J=5 stochastic samples of G0             cost-matched control for S2
S5  oracle best single generator              upper bound
```

**S4 matters.** Without it, any S2 gain is confounded with simply having five hypotheses instead of
two. S2 must beat S4, not just S0.

---

## 4. Beta must be re-swept

Ensemble size changes the number of fused rankings, which changes the fused-score range, which
changes what `beta` means. Range normalization makes it scale-invariant in principle, but the score
distribution still shifts.

```
For each arm, sweep beta in {0.2, 0.4, 0.6, 0.8, 1.0, 1.5}
Report the peak per arm and the rerank_share column
  rerank_share = mean over facts of
     (beta * range(consensus)) / range(normalized fused score)
```

Compare arms **at their respective optima**, not all at `beta=0.6`. Comparing a tuned baseline
against an untuned variant is how you manufacture a false negative.

---

## 5. Metrics and reads

R@10, R@50, R@200, MRR. Paired per fact, bootstrap CIs at the **source-context** level, 2,000
iterations. Table and text reported separately.

Primary, table subset:

```
a)  S2 - S0   does specialization beat the frozen baseline?
b)  S2 - S4   does specialization beat cost-matched stochastic sampling?
              this is the real test
c)  S1 - S0   does it work at 3 generators, i.e. cheaper?
d)  S2 - S5   how much of the best-single-generator oracle does
              fusion capture?
```

Also report, per generator:

```
solo R@10 and MRR                     which readings are individually useful
selection frequency under agree()     which the verifier prefers
mean resolved-dimension count         are specialists appropriately cautious
pairwise neighborhood Jaccard         is variation structural or cosmetic
```

That per-generator table is worth as much as the ensemble result. If one generator is solo-strong
and the rest never contribute, the finding is "one better prompt," not "specialization."

---

## 6. Decision rules, fixed in advance

| Observation | Reading | Action |
|---|---|---|
| `S2 - S4` > 0, CI excludes zero, on R@10 or MRR | specialization beats cost-matched sampling | adopt; §4.2 becomes specialized generators and the multi-agent framing is earned |
| `S2 - S4` positive, CI spans zero | suggestive, underpowered at 30 table contexts | report as an ablation, keep stochastic sampling as the method |
| `S2 - S4` ≤ 0 | specialization does not help | keep the current §4.2; add one ablation row; the "On functional specialization" paragraph stands as written |
| `S1 ≈ S2` | the extra two specialists add nothing | use 3 generators if adopting |
| one generator dominates solo and others rarely contribute | it is a prompt improvement, not an architecture | fold the better prompt into G0 and drop the rest |
| specialists resolve *more* dimensions than G0 on average | they are guessing outside their remit | tighten the abstention instruction and re-run before reading anything else |

---

## 7. Prior warning to record in the output

The coverage pilot measured two interventions on hypothesis diversity, and **both reduced
coverage**: a diversity-directed prompt reached 0.664 and a per-dimension assignment 0.682, against
0.735 for plain stochastic sampling at table `K=200`. In each case single-hypothesis recall fell
5–8 points — diversity was purchased by degrading each hypothesis.

Specialization is not the same manipulation, since each generator is asked for its best reading
rather than a different one. But it is the same family, so treat a null or negative result as the
expected outcome rather than as a bug in the run. Record this note in `metrics.json`.

---

## 8. Outputs

```
specialization.csv        arm x modality x metric, paired CIs, at per-arm optimal beta
generator_solo.csv        per-generator solo performance, selection frequency,
                          resolved-dimension count, pairwise Jaccard
beta_sweep_multigen.csv   full sweep, all arms, with rerank_share
hypotheses.jsonl          per fact per generator: dimensions raw + normalized,
                          both rendered queries
retrievals.jsonl          per fact per generator per rendering: ranked candidate
                          ids, gold rank
metrics.json              selected configuration, decision-rule outcome, and the
                          §7 prior warning
```

---

## 9. What this decides

Only whether §4.2 of the paper describes stochastic sampling or specialized generators. It does not
affect the frozen AGS test-split run, which should proceed independently — this is off the critical
path and its outcome can only upgrade the method section, never block it.
