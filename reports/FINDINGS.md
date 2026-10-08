# Findings log

A running record of results as each component finishes, for the report and the paper (C18).
Every number here traces to a file in `reports/tables/`. Where a result comes from MLflow runs, the file
lists their run IDs.

---

## C2: bridge check (evidence for H8, extractor sensitivity)

Source: `reports/tables/bridge_check.csv` and `bridge_check_summary.csv`. LycoS18 and NF-CSE-CIC-IDS2018-v2 come
from the same CSE-CIC-IDS2018 PCAPs, processed by different flow extractors.

- On **benign** traffic, 15 of the 18 mapped core features agree (KS ≤ 0.125) once units are harmonised.
- Three features disagree and were dropped:
  - `duration_s` (KS 0.91): NetFlow-v2 records duration 0 for 82–94% of flows.
  - `init_win_fwd` and `init_win_bwd` (KS 0.38 and 0.45): initial window vs maximum window are different quantities.
- **Within attack families the same traffic gives very different flows.** The median per-family KS of packet and
  byte counts is 0.78–0.98. LycoS18 has 1.80M DoS Hulk flows, while NF-CSE has 0.43M from the same PCAPs.

## C3: deduplication

Source: `reports/tables/clean_*.csv`.

- **NetFlow-v2 records are 80–90% exact duplicates.** NF-CSE goes from 18.9M rows to 2.0M, and NF-ToN from 16.9M to
  1.85M. Some attack families nearly vanish: NF-CSE Bot drops from 143,097 rows to 472 unique.
- **Label conflicts are mostly noise in NF-CSE.** 14,056 vectors carry both labels, typically Benign plus a few
  `Infilteration` copies. In LycoS18, 52 vectors carry both FTP-Patator and DoS Slowhttptest.

## C4: adversarial validation (input to H3)

Sources:
- `reports/tables/advval_summary.csv`, with MLflow run IDs per row;
- the curves in `reports/tables/advval/`;
- the figures `reports/figures/advval_{cic77,nf43,core}.png`.

Method: a LightGBM domain classifier, 5-fold CV, on 100k train-split rows per dataset, with 3 seeds. Feature
importance is gain. At each step the top feature is removed, up to 10 times or until AUC < 0.7.

1. **On native features every pair of datasets is almost perfectly separable, and stays so.**
   - LycoS17 vs LycoS18 (cic77, benign): AUC 0.9993 ± 0.0000, and **0.9984 after removing 10 features**.
   - Every nf43 pair: AUC ≥ 0.9999 before and after.
   - No feature carries more than ~24% of the classifier's gain, so the lab signature is spread over many
     features. This is the Build Guide's anticipated "finding, not a bug": removing the top-k features cannot
     make the labs indistinguishable. Expect C7's ablation to explain only part of the collapse.
2. **What tells the labs apart.**
   - cic77: the ECN flags `flag_ece` and `flag_cwr` (the 2017 and 2018 hosts negotiate ECN differently), then
     `dst_port`, then minimum inter-arrival times.
   - nf43: TTL (`min_ttl`, `max_ttl`), which reflects each lab's operating systems and hop counts; minimum packet
     length; `l4_dst_port`; retransmission counts.
   - In cic77 the removal order is identical across all 3 seeds: the first 7 steps on benign traffic, all 10 on
     all traffic.
3. **`dst_port` is the leading lab-telling feature.** It ranks first in 12 of the 28 (pair, subset) cases and is
   in the top 3 in 18. This supports keeping it flagged and ablating it explicitly in C7.
4. **On the 15-feature core track, removal does bite.** Initial AUC is 0.96–1.00. After 9–10 removals (5–6
   features left), it falls to 0.62–0.91, and 6 of the 20 curves cross 0.7.
5. **The extractor alone separates the data about as well as a different network.** On core benign traffic:
   - LycoS18 vs NF-CSE-CIC-IDS2018-v2 (same PCAPs, different extractor): AUC **0.964**.
   - LycoS17 vs LycoS18 (same extractor, different year and network): AUC **0.965**.

   This is direct evidence for H8.

Hand-off: `reports/tables/advval_lab_features.json` holds the consensus removal order per track, pair and subset
(mean position across seeds). C7 ablates its top-k; C8 checks its drift explanations against it.

## C5: harness acceptance run (LycoS17 → LycoS18, cic77, seed 0, dev subsample)

Source: the MLflow experiment `train` (one run per model, tags model / track / source / seed). These are
single-seed sanity numbers. C6 reports mean ± std over 3 seeds.

| Model | Within: FPR @ thr | Within: DR @ thr | Within MCC | Cross: FPR @ thr | Cross: DR @ thr | Cross oracle FPR@95%DR | Cross PR-AUC |
|---|---|---|---|---|---|---|---|
| LDA | 0.0055 | 0.827 | 0.876 | 0.0504 | 0.0024 | 0.870 | 0.296 |
| DT | 0.0031 | 0.908 | 0.935 | 0.0234 | 0.0284 | 0.891 | 0.344 |
| RF | 0.0021 | 0.869 | 0.913 | 0.0000 | 0.0000 | 1.000 | 0.456 |
| XGB | 0.0013 | 0.830 | 0.891 | 0.0000 | 0.0000 | 0.849 | 0.538 |
| MLP | 0.0013 | 0.844 | 0.900 | 0.0008 | 0.0006 | 0.839 | 0.277 |
| AE | 0.0395 | 0.939 | 0.858 | 0.6810 | 0.4666 | 0.858 | 0.267 |
| TabNet | 0.0025 | 0.857 | 0.904 | 0.0105 | 0.0007 | 0.414 | 0.472 |

- **The sanity gate passes:** within-dataset MCC on LycoS17 is 0.86–0.94, in line with Cantone et al.'s 94.63%
  within-dataset average.
- **The collapse is total for the supervised models.** At the frozen source threshold they catch 0–3% of LycoS18
  attacks (Cantone et al.'s worst pair is this same LycoS17 → LycoS18, MCC 10.83%). This is not just
  miscalibration: even the oracle threshold needs FPR 0.41–1.00 to catch 95%, so the ranking itself breaks.
  TabNet's ranking survives best (oracle FPR 0.41).
- **The autoencoder fails the other way.** It flags 68% of LycoS18 benign traffic, because a new network's normal
  traffic looks anomalous. The supervised models stay quiet and miss the attacks; the anomaly detector raises
  floods of false alarms.
- **Within-dataset DR at the frozen threshold is 0.83–0.94, not 0.95.** LycoS17's time-block split puts
  different time blocks, and so different attack bursts, into val and test. The attack mix shifts (Recon is 21.9%
  of val attacks vs 15.1% of test; DoS 49.3% vs 53.0%), and bursts of the same family can differ too. So even inside
  one dataset, a threshold set on one period does not hold its DR on another. This is a small-scale preview of H4,
  and C7 can separate the two causes.

## C6: results matrix (H1, H2, comparison with Cantone et al.)

Sources: `reports/tables/matrix_{cic77,nf43,core,cic_orig,h2_control}.csv`, where every cell lists its MLflow
run IDs, plus `matrix_long.csv`, `h1_checks.csv`, `cantone_comparison.csv` and `h2_original_vs_corrected.csv`.
Figures: `reports/figures/matrix_*`, `within_vs_cross_*`.

Coverage: 7 models × 3 seeds on cic77, nf43, cic_orig and the H2 control, and RF/MLP/LDA/XGB on core. This is
more than the Build Guide's minimum, with 0 planned runs missing. All runs use the laptop dev subsample of at most
2M rows per split. TabNet uses patience 10: patience 5 left 2 of 3 NF-UNSW seeds untrained, which was visible on
source validation (PR-AUC 0.21–0.31 vs 0.93), so the fix uses no target information. All TabNet runs were redone.

### H1 (collapse): supported, with a caveat about how it shows

| Track | Within MCC | Cross MCC | Within DR @ thr | Cross DR @ thr | Within oracle FPR@95%DR | Cross oracle FPR@95%DR |
|---|---|---|---|---|---|---|
| cic77 | 0.910 | 0.161 | 0.909 | 0.264 | 0.022 | 0.646 |
| nf43 | 0.874 | 0.086 | 0.951 | 0.357 | 0.056 | 0.923 |
| core | 0.864 | 0.120 | 0.928 | 0.271 | 0.099 | 0.699 |

(Means over cells and models.)

- **134 of 136 cross cells have lower MCC and higher oracle FPR than the within cell of the same model and source.**
  The two exceptions are both LDA on the bridge pair LycoS18 ↔ NF-CSE-CIC18 (core track), i.e. the same traffic
  extracted by two different tools.
- **The frozen-threshold FPR rises in only 91 of 136.** Supervised models mostly fail *silently* on a new network:
  the threshold set on source validation flags almost nothing, so FPR stays low while DR collapses. Random forest
  is the extreme case: its median cross-dataset DR is 0.01%. The autoencoder fails the other way, flagging 92–100%
  of benign traffic on every nf43 cross pair. **So H1 should be stated as "detection and ranking collapse", not
  "FPR rises".** That is why FPR is always reported together with DR.
- **Transfer is asymmetric.** Training on LycoS18 transfers partly to LycoS17 (MCC 0.41–0.57 for DT, XGB, TabNet,
  LDA and MLP). Training on LycoS17 transfers to nothing (MCC ≤ 0.014).
- **Diverse training data transfers best.** On core, RF trained on NF-ToN, the most attack-diverse dataset, reaches
  MCC 0.851 on LycoS18 (DR 94%, FPR 7%), across extractor families. On nf43, the best cross cell is MLP NF-ToN →
  NF-CSE18 (MCC 0.756).

### Comparison with Cantone et al. (2024)

| | Ours (LDA/DT/RF/XGB, cic77) | Cantone et al. |
|---|---|---|
| Within-dataset average MCC | 0.933 | 0.946 |
| Cross-dataset average MCC | 0.169 | 0.294 (12 pairs; ours: the 2 LycoS pairs) |
| LycoS17 → LycoS18 (avg of 4 models) | −0.028 | 0.108 (their worst pair) |
| LDA LycoS18 → LycoS17 | 0.559 | 0.604 (their best single cross result) |

- **Their key LycoS results reproduce.** LycoS17 → LycoS18 is the worst direction, and LDA LycoS18 → LycoS17 is a
  strong cross result (0.56 vs 0.60).
- **"LDA generalises best" does not hold as an average.**
  - On cic77, the mean cross MCC by model is MLP 0.28, XGB 0.25, LDA 0.22, DT 0.21, TabNet 0.19, RF 0.00, AE −0.02.
  - On nf43, LDA is near the bottom (0.008); TabNet (0.195) and DT (0.176) lead.
  - On core, LDA ties with the MLP for best (0.156).

  LDA's best cross cell is strong, but it does not generalise best overall.

### H2 (label inflation): not supported once the split is controlled

The H2 control re-splits LycoS17 at random (`lycos17__random`: the same 1,633,215 rows), so it is compared with the
original CIC-IDS2017 like for like. Within-dataset results:

| Model | MCC original | MCC corrected, random split | MCC corrected, time blocks | PR-AUC original − corrected (random) |
|---|---|---|---|---|
| DT | 0.979 | 0.971 | 0.935 | −0.0007 |
| RF | 0.970 | 0.967 | 0.914 | −0.0001 |
| XGB | 0.970 | 0.966 | 0.891 | −0.0001 |
| MLP | 0.970 | 0.966 | 0.901 | −0.0001 |
| TabNet | 0.971 | 0.967 | 0.894 | −0.0002 |
| LDA | 0.719 | 0.936 | 0.876 | −0.079 |
| AE | 0.787 | 0.864 | 0.834 | −0.038 |

- **On like-for-like splits, the original release is at most 0.003–0.008 MCC higher** for the five strong models.
  PR-AUC differs by at most 0.0007. For LDA (−0.22 MCC) and the AE (−0.08), the original release is clearly *worse*,
  plausibly because CICFlowMeter's miscomputed features hurt linear and reconstruction models. There is no evidence
  of meaningful label-error inflation of within-dataset scores.
- **The split scheme matters much more than the release.** The time-block split alone costs the corrected release
  0.03–0.08 MCC. The uncontrolled comparison would have mistaken this for a large H2 effect.
- **Remaining confound:** the two releases use different extractors (CICFlowMeter vs LycoSTand), so this is
  "original vs corrected release", not label errors in isolation.

## C8: drift monitor (H5)

Sources: `reports/tables/drift_h5.csv`, `drift_h5_by_track.csv`, `drift_eval_long.csv` (MLflow run ID per row),
`drift_vs_retraining.csv`, `drift_explanations.csv`; figure `reports/figures/drift_switch_cic77_lycos17_lycos18.png`
(Mid-Sem v1).

Setup:
- 8 ordered pairs (cic77 both directions, all 6 nf43 pairs) × 3 seeds, using each source's C5 MLP.
- The reference is 20,000 source-validation flows; windows hold 5,000 flows.
- **null:** held-out source-validation flows, which are exchangeable with the reference.
- **later:** the source's test split.
- **switch:** 10 null windows, then 10 target windows.
- **ramp:** the target share rises from 0 to 100% over 20 windows.

| Detector | False alarms / 100 null windows | Switch detected | Switch delay (flows) | Ramp: target share at first alarm |
|---|---|---|---|---|
| KS (Bonferroni) | 0.2 | 100% | 5,000 | 6% |
| KS + effect size (D ≥ 0.1) | 0.0 | 100% | 5,000 | 18% |
| MMD (α 0.05) | 2.9 | 100% | 5,000 | 8% |
| ADWIN on confidence | 0.0 | 100% | 6,042 | 34% (96% of ramps detected) |
| Combined (≥ 2 of KS-effect, MMD, ADWIN) | 0.0 | 100% | 5,000 | 18% |
| Combined + ATC cost trigger ("recommend") | 0.0 | 67% | 5,313 | 19% (71% of ramps) |

- **Every detector catches a sudden network switch within the first window (5,000 flows), with no false alarms
  for KS-effect, ADWIN and the combined rule.** MMD's 2.9% false-alarm rate is in line with its α of 0.05.
- **Sensitivity on gradual drift ranks KS > MMD > KS-effect = combined > ADWIN.** ADWIN reacts only once the
  model's confidence moves, at about a third of the traffic.
- **The same network later in time is real drift.** On LycoS17's test time blocks, KS alarms on 51% of windows and
  the combined rule on 45%, consistent with the within-dataset DR drop seen in C5/C6 (H4 preview). On the randomly
  split NF datasets the later stream behaves like the null (0–3%).
- **The ATC cost trigger misses a third of the switches.** This is exactly the "silent" failure from C6: on, e.g.,
  NF-ToN → NF-CSE18 the model stays confident while its predicted attack share falls from 0.57 to 0.19, so ATC
  estimates no error rise. The predicted attack share is the better label-free signal there; feed it to C11.
- **H5 is supported under the stated cost assumptions** (`configs/drift.yaml`: 100k flows/hour, 1 unit per wrong
  decision, 2,000 per adaptation). With one network switch per day, the combined monitor acts once, costs 2,000
  per day and leaves ~5,000 flows unhandled. Retraining every 1M flows costs 4,800 per day with ~500,000 flows
  unhandled; every 50k flows costs 96,000 per day with ~25,000 unhandled.
- **The explanations agree with C4.**
  - On nf43, all 5 of the top-5 drifted features after a switch are among the C4 lab-telling features for that
    pair, in every pair and seed.
  - On cic77, 2 of 5 match: the ECN flags lead. Example message: "flag ece is zero in 60% of flows vs 100% in
    training; flag cwr is zero in 60% of flows vs 100% in training; forward inter-arrival time min is 54.9x higher
    than in training".
- **Evaluation bug caught and fixed:** windows were first built in file order. NF-ToN files group flows by class,
  which made ADWIN, the only order-sensitive detector, fire on 100% of NF-ToN null windows. Windows are now
  randomly ordered, and ADWIN's null false-alarm rate is 0.

## C7: decomposition (H3) and time drift (H4)

Sources:
- `reports/tables/c7_long.csv` (333 runs, MLflow experiment `c7`, tag `study=c7`), `c7_shares.csv`,
  `c7_norm_shares.csv`;
- `h4_timedrift.csv`, `h4_summary.csv`, `h4_by_day.csv`;
- figures `reports/figures/c7_share_mcc.png`, `h4_benign_fpr_over_time.png`.

### H3: removing or normalising lab-telling features explains ~0% of the collapse

Setup:
- RF, XGB and the MLP are retrained without each pair's top-k C4 lab-telling features (benign ranking, k ∈
  {1, 3, 5, 10}), with the C6 model configs, for 8 ordered pairs × 3 seeds.
- Share of collapse explained = (cross_k − cross_0) / (within_0 − cross_0) on MCC; analogous on oracle FPR and
  PR-AUC; reported raw.

Results:
- **The median share is 0.000 (IQR −0.007 to 0.007) over all 96 (model, pair, k) cells.** Only 3 cells exceed
  0.2, all the MLP on NF-UNSW → NF-CSE18 (0.28–0.34); 8 are below −0.2. Oracle-FPR and PR-AUC shares tell the
  same story.
- **Removal costs nothing within-dataset** (median change in within MCC 0.000). The lab-telling features are not
  needed to detect attacks in their own lab, and removing them does not make the model transfer.
- **The Build Guide's frozen-threshold FPR formula is not usable:** its denominator FPR_cross − FPR_within is ≤ 0
  in 48 of 96 cells, because models fail silently (C6). It is kept in `c7_shares.csv` as `share_fpr_thr`.
- **Per-domain normalisation is worse than doing nothing.** A rank transform per domain, with the target's fit on
  its unlabelled pool, takes mean cross MCC from 0.104 to −0.003. It destroys the partial transfer that existed:
  LycoS18 → LycoS17 MLP falls from 0.572 to 0.102 and XGB from 0.497 to 0.000. Absolute values carry attack
  information that per-domain ranks erase.
- **Interpretation (H3 fallback: "0% explained" is a publishable finding).** Together with C4, where the domains
  stay separable after 10 removals, the lab signature is spread across the whole feature space. Within-family
  attack flows also differ between labs (C2 bridge check). So the collapse is not caused by a few spurious
  features that could be dropped.

### H4: gradual drift raises false alarms far less than a sudden switch

Setup:
- Models are trained on LycoS17's Monday + Tuesday (train-split rows; threshold on Mon–Tue val-split rows).
- They are evaluated hour by hour on Wednesday–Friday and on LycoS18 (the sudden switch).
- Rates are pooled over flows. Hours with < 200 benign flows are not plotted: the first hour of a capture day can
  hold 2–5 flows.

| Autoencoder (benign-only, so its alarms measure "unlike training-period normal") | Benign FPR | Benign KS max vs Mon–Tue |
|---|---|---|
| Mon–Tue test | 1.63% | 0.057 |
| Wednesday / Thursday / Friday | 1.25% / 1.30% / 1.38% (worst hour 3.3%) | 0.100 (later days) |
| LycoS18, sudden switch | **58.3%** | 0.391 |

- **H4 is supported.** Over three days the features drift measurably (KS 0.057 → 0.100), but the anomaly
  detector's false alarms stay flat (1.3%). A different network raises them 45× (58.3%) with a feature shift
  four times larger.
- The supervised models trained on Mon–Tue are brute-force detectors. They stay silent on every later day and on
  LycoS18 (benign FPR ≤ 0.04%, DR on the unseen attack families ≈ 0). This is the C6 "silent" failure, so their
  benign FPR cannot show drift in either direction.
- Confound (Build Guide): each CIC-2017 day runs a different attack. That is why benign-only measures are the
  primary H4 evidence and DR over time is reported only with this caveat.
- Correction to an earlier interim report: a "Wed 1.3% → Thu 5.8% → Fri 11.3%" trend came from averaging hourly
  rates without weights. It was driven by a 2-flow 08:00 block, and the pooled rates above replace it.

## C9: adapters (before / after, first evidence for H6)

Sources: `reports/tables/c9_long.csv` (324 rows, MLflow experiment `c9`, config v2), `c9_summary.csv`,
`reports/figures/c9_before_after.png`.

Setup:
- Each source's C5 bundle is adapted using a 200k-flow unlabelled target pool (target train split) and, for
  few-shot, labels bought from that pool.
- Scored on the target test split at the adapted bundle's threshold, and back on the source test split
  (forgetting).
- MLP: 12 adapter settings × 8 pairs × 3 seeds. XGBoost: scaling + few-shot × 3 pairs × 3 seeds.
- Hyperparameters were fixed before any target result (CORAL λ ∈ {0.1, 1, 10}, all reported).

| MLP, mean over pairs and seeds | cic77 MCC (before 0.280) | nf43 MCC (before 0.141) | Labels |
|---|---|---|---|
| scaling | 0.293 | 0.060 | 0 |
| **AdaBN** | **0.535** | 0.031 | 0 |
| Tent | 0.488 | −0.037 | 0 |
| CORAL λ 0.1 / 1 / 10 | 0.381 / 0.366 / 0.445 | 0.047 / 0.035 / 0.035 | 0 |
| DANN | 0.422 | 0.019 | 0 |
| few-shot random 50 / 200 / 1000 | 0.733 / 0.745 / 0.721 | 0.434 / 0.439 / 0.538 | 50–1000 |
| few-shot 200, uncertainty / drift selection | 0.325 / 0.274 | 0.317 / 0.146 | 200 |

| XGBoost | cic77 MCC (before 0.248) | nf43 MCC (before 0.509) | FPR change | Source MCC change |
|---|---|---|---|---|
| few-shot random 50 / 200 / 1000 | 0.649 / 0.848 / **0.915** | 0.521 / 0.840 / **0.940** | ≤ +0.2 pt | cic77 ≈ 0; nf43 −0.45 to −0.59 |
| scaling | 0.000 | −0.007 | — | −0.92 |

- **Label-free adaptation works on cic77 and fails on nf43.**
  - On LycoS17 ↔ LycoS18, AdaBN (no labels, no gradients, < 1 s) roughly doubles MCC (0.28 → 0.54) for +1.2
    points of FPR. Tent, CORAL and DANN follow.
  - On the NetFlow pairs every label-free adapter makes the target *worse*, and AdaBN also costs 0.45 MCC on the
    source. Re-normalising does not fix a shift in what the features mean (different labs, attack families,
    extractor settings).
- **Few-shot labels are the reliable lever.**
  - 50 random labels already lift the MLP to MCC 0.73 on cic77 and 0.43 on nf43.
  - XGBoost retrained with 1,000 labelled target flows (upweighted to 20% of the training weight) reaches 0.915
    and 0.940 with no FPR increase.
  - Caveat: the MLP's threshold is re-picked on a tiny budget (13–58 attacks on average for 50–200 flows), which
    costs 3–26 FPR points. XGBoost, with its upweighted retraining, does not pay that.
- **Selection rule matters, and random wins.** With 200 labels both alternatives fall far below random selection,
  for different reasons:
  - drift-guided selection buys almost no attacks (0.2 per 200 flows on cic77, 23.5 on nf43), because the most
    "unlike the source" flows are benign;
  - uncertainty selection buys plenty of attacks (78–96 per 200) but only flows sitting at the old decision
    boundary, which do not represent the target.
- **Per-domain scaling** (moment matching onto the source scale) does nothing for the MLP and destroys XGBoost
  (MCC → 0). Tree splits on absolute thresholds do not survive re-scaling. This is consistent with C7.
- **Tent's guard** stopped adaptation early in 14 of 24 runs. A first guard version, measured against the
  silently failing model's own near-zero attack rate, blocked every useful step. It now follows the Build Guide
  and uses the source attack rate.
- **H6 is partly supported.** On the same-extractor pair, label-free test-time methods (AdaBN, Tent) beat
  CORAL/DANN at zero label cost. Across labs (nf43) no label-free method helps, and a few hundred labels beat
  everything. C12 (Reptile) completes H6.
- **Fixed during C9:** the scaling input map overflowed to NaN scores on NF-CSE18 → NF-UNSW, where a feature is
  nearly constant on the target. The map is now clipped to the source's range, and C9 was re-run in full (config
  v2).

## C10: adaptation logs

Sources: `data/logs/adapt_log.parquet` (5,600 rows; one MLflow run per pair in experiment `c10`),
`reports/tables/c10_summary.csv`, `reports/figures/c10_sanity.png`.

Setup:
- 8 ordered pairs × 50 drift windows, drawn from each target's train split; the test splits are untouched.
- Window kinds: random, family-mix, source/target blend, and LycoS17 time slices. Window sizes 5k–20k.
- Each window is halved: an unlabelled adaptation half and a held-out outcome half.
- Drift features: the C8 monitor on the adaptation half, before any action.
- 14 actions per window: wait, scaling, AdaBN, Tent, few-shot 50/200/1000, CORAL ×3 and DANN (trained once per
  pair and seed), XGBoost wait and few-shot 200/1000.

Results:
- **The best action per window**, over all actions and judged by MCC gain against the deployed MLP:
  - with no label cost, a labelled action wins 98–99% of windows. XGBoost few-shot with 1,000 labels alone wins
    72% (cic77) and 86% (nf43); label-free adapters win 0% and 0.7%;
  - at a label cost of 2e-4 MCC per label, XGBoost few-shot with 200 labels wins 56% / 50%, label-free adapters
    win 8% / 7%, and waiting wins 3% / 8%.

  (An earlier draft said "90–94%". That came from `best_share`, which ranks actions only within one model; the
  column is now named `best_within_model_share`.)
- **Mean MCC change** — cic77: AdaBN +0.125, CORAL λ 10 +0.094, MLP few-shot +0.35 to +0.40, XGBoost few-shot
  +0.47 / +0.54. nf43: every label-free adapter −0.02 to −0.13, MLP few-shot +0.21 to +0.27, XGBoost few-shot
  +0.45 / +0.57.
- **`delta_fpr` (the Build Guide's outcome) points the wrong way for the most useful actions.** MLP few-shot has
  `delta_fpr` −0.07 to −0.17 (FPR rises) while MCC improves by +0.2 to +0.4. The logs therefore keep MCC, DR and
  PR-AUC alongside FPR.
- Tent and AdaBN gain less here than in C9 (cic77: Tent ~0 vs +0.21). Each window gives them only 2.5k–10k flows
  instead of 200k, and blend windows still contain source traffic.

## C11: adapter selector (H6 extension)

Sources: `reports/tables/c11_lopo.csv`, `c11_summary.csv`, `reports/figures/c11_regret.png`, MLflow `c11`;
deployable selector in `models/selector/`.

Setup:
- The deployed model is the source MLP.
- The selector is a 5-model bootstrap LightGBM ensemble. Input: the 10 label-free drift features plus the action.
  Output: the action's utility.
- Utility = MCC gain − label_cost × labels; label_cost = 2e-4 per label (1,000 labels cost 0.2 MCC).
- Evaluation is leave-one-pair-out over the 8 pairs.
- Baselines: always-Tent, the best single fixed action (chosen on the training pairs), random, and the oracle.

| Method (label cost 2e-4) | Mean regret vs oracle | Top-1 agreement | Mean realised utility |
|---|---|---|---|
| oracle | 0 | 1.00 | 0.449 |
| **best fixed: XGBoost few-shot with 200 labels (chosen in every fold)** | **0.043** | 0.51 | 0.406 |
| selector | 0.067 | 0.42 | 0.382 |
| random | 0.400 | 0.07 | 0.049 |
| always-Tent | 0.466 | 0.04 | −0.017 |

- **The selector does not beat the best fixed action.** Regret is 0.067 vs 0.043; it wins on 4 of 8 held-out
  pairs, one of them a tie (0.0835 vs 0.0836). This holds at every label cost tested (0, 1e-4, 2e-4, 5e-4) and
  with the FPR utility. Its predictions rank actions reasonably (Spearman 0.45–0.65 between predicted and actual
  gain), and both learned and fixed choices are far better than random or always-Tent.
- **Why:** one action dominates almost everywhere, because labelled retraining is that much stronger than any
  label-free adapter across these pairs. Window-level drift features add little beyond "use labels". The
  selector's errors concentrate on the pair least like its training pairs (NF-UNSW → NF-CSE18, regret 0.142 vs
  0.028).
- **What it does add:** it waits when nothing is predicted to beat waiting by the margin (wait share 9% at label
  cost 2e-4, 13% at 5e-4), and it gives an uncertainty for the dashboard. At label cost 0 it matches the fixed
  rule (regret 0.005 vs 0.004).
- Honest summary for the paper: with full-information logs, a learned selector is not needed when one adapter
  dominates. It becomes useful only when the action ranking varies with the drift.

## C12: Reptile meta-learning (H6)

Sources: `reports/tables/c12_long.csv` (24 folds × 33 rows, MLflow `c12`, config v2), `c12_summary.csv`,
`c12_paired.csv`, `reports/figures/c12_h6.png`; meta-models in `models/reptile/<track>/<held-out>-s<seed>/`.

Setup:
- Leave one target out: nf43 (3 targets) and core (5 targets), 3 seeds.
- θ₀ = the C5 MLP trained on the pooled other datasets. Reptile meta-trains from θ₀ on tasks of the form
  (window, labelled support set of 50/200/1000), with 30% synthetic rescaled domains.
- Inner loop = AdaBN, then 8 steps.
- Inner optimiser and outer step: chosen per fold by meta-validation on the held-in domains' validation splits.
  SGD lr 0.01 / ε 0.1 was chosen in 92% of folds.
- The fair baseline is θ₀ fine-tuned with the identical steps on the identical support sets (5 draws per budget).

| Mean target MCC | none | AdaBN only | 50 labels | 200 labels | 1000 labels |
|---|---|---|---|---|---|
| core: fine-tune θ₀ | 0.265 | 0.442 | 0.699 | 0.739 | 0.734 |
| core: Reptile | — | 0.408 | 0.695 | 0.752 | 0.726 |
| nf43: fine-tune θ₀ | 0.407 | −0.138 | 0.464 | 0.588 | 0.591 |
| nf43: Reptile | — | −0.126 | 0.403 | 0.521 | 0.492 |

- **H6, meta-learning part: not supported.**
  - On core, Reptile equals plain fine-tuning from the same start: paired difference −0.004 / +0.013 / −0.008 at
    50 / 200 / 1000 labels; it wins 47–49% of paired comparisons.
  - On nf43 it is worse by 0.06–0.10 MCC (wins 29–49%).
  - On the held-in validation domains Reptile did adapt better (PR-AUC 0.93 vs 0.90). That advantage does not
    carry over to a lab it has never seen.
- **Labels are what matter.** Both labelled methods lift the pooled model strongly (core 0.27 → 0.74; nf43
  0.41 → 0.59 with 200 labels). Zero-shot adaptation does not: AdaBN-only gives the same result from θ₀ and from
  the meta-model, and it hurts on NetFlow (MCC −0.13).
- **Bug found and fixed (v1 discarded).** The first run used Adam inner steps (lr 1e-3) with outer step 0.1. Meta-
  training then moved θ further from θ₀ (‖Δ‖ 32.5) than θ₀'s own norm (31.6), and the meta-model scored ROC-AUC
  0.52 on its own training domain. Every v1 Reptile number was a degenerate model. v1 also picked zero-shot
  thresholds without AdaBN. v2 selects the inner settings per fold and picks thresholds after AdaBN; ‖Δ‖/‖θ₀‖ is
  now 0.05–0.12.
- **Overall H6 verdict (C9–C12):**
  - Test-time adaptation (AdaBN/Tent) helps only within one extractor family (cic77) and hurts across labs.
  - CORAL/DANN are weaker than AdaBN on cic77 and also hurt on nf43.
  - Meta-learned fast adaptation adds nothing over fine-tuning with the same labels.
  - The cheapest reliable recovery is a few hundred labelled target flows: an XGBoost retrained with them reaches
    MCC 0.92–0.94 (C9).

### C10 v2 / C11 re-run with Reptile actions (9 Oct 2026; supersedes the C10/C11 numbers above)

- **C10 v2:** 6,500 rows. The nf43 windows gain 3 Reptile actions: switch to the meta-model trained WITHOUT that
  target, AdaBN, then 8 steps on 50 / 200 / 1000 bought labels. Mean MCC change on nf43: Reptile +0.18 / +0.22 /
  +0.20, against XGBoost few-shot +0.45 / +0.57 and MLP few-shot +0.21 to +0.27. At label cost 2e-4, Reptile-200
  and Reptile-50 are the best action in 7.7% and 6.0% of nf43 windows; XGBoost few-shot 200 is best in 45%.
- **C11:** the conclusion is unchanged. Mean leave-one-pair-out regret:

| Label cost | 0 | 1e-4 | 2e-4 | 5e-4 |
|---|---|---|---|---|
| best fixed (XGBoost few-shot 200) | 0.004 | 0.037 | **0.050** | 0.061 |
| selector | 0.006 | 0.065 | **0.077** | 0.082 |
| random | 0.469 | 0.422 | 0.411 | 0.424 |
| always-Tent | 0.572 | 0.505 | 0.472 | 0.424 |

  The selector beats the fixed action on 3 of 8 held-out pairs (LycoS18 → LycoS17, NF-CSE18 → NF-UNSW, NF-ToN →
  NF-UNSW).

## C13: demo data (NFStream flows, demo model, demo.pcap)

- **Extractor (frozen).** `configs/nfstream.yaml`, hash `497b9ed951` (in `configs/nfstream.lock`; `extract()` refuses
  to run if they differ). Settings: idle timeout 120 s, active timeout 1800 s, statistics on, no nDPI.
  `xnids.live.extract.TCPTermination` ends a TCP flow on RST, or on the ACK after both FINs.
  - **Bug found before any model was trained:** without the plugin, NFStream merged about 12.5 port-reusing DoS-Hulk
    connections into one flow (14,080 flows, median 935 s). With it, the count is 185,549, in line with LycoS17's
    158,988. The other families now match LycoS17 to within 1.3%: DDoS 94,522 vs 95,683, PortScan 159,275 vs
    160,106, GoldenEye 6,753 vs 6,765, Bot 735 vs 735. The 2017 PCAP clock is UTC, as are the LycoS rules, and the
    rules found every attack window on the exact attacker/victim pair with no reversed flows.
- **CSE-CIC-IDS2018 without the 36 GB archive.**
  - `scripts/fetch_cse18_pcaps.py` reads the zip's central directory with HTTP range requests, downloads only 15
    of the 445 members of Fri-16-02-2018 (8.8 GB unpacked, against 36 GB zipped for the whole day) and checks each
    CRC-32.
  - **Clock offset measured:** local = UTC−4 (AST). The SlowHTTPTest source appears 14:12–15:05 UTC for a
    scheduled 10:12–11:08, and Hulk starts 17:45:27 UTC for a scheduled 13:45. The UTC−5 assumed in the plan was
    wrong.
  - Three of the 13 workstation captures are corrupt partway through, upstream; the CRC matches the archive.
- **The 2018 "DoS-SlowHTTPTest" never reached a web server.** All 105,550 flows go to **port 21**. Each is a client
  SYN answered by a server RST (no SYN-ACK; 2 packets, 0 ms). The labels follow the official schedule, but these
  are refused connections, which look like a port scan. The demo model flags 100% of them, for that reason.
- **Data.**
  - nfs17 has 1,556,428 flows. Dedup removes 44%, against about 10% for LycoS17. NFStream's millisecond
    timing makes short flows (DNS, 2–4-packet TCP) collide, and 155k of 159k PortScan probes are exact repeats.
  - The split is by time block, with Wed 13:30–13:55 UTC held out (146,661 rows) as replay segment A.
  - nfs18 has 2,089,815 flows, 92% of them DoS-Hulk, from a 13-minute burst. Its split is stratified: time
    blocks had left a benign-only test split, so that first attempt was discarded before use.
- **Demo model** (MLP, 59 NFStream features, full nfs17 train split, 3 seeds, ~4 s per seed on GPU):
  - Within nfs17: FPR 1.2–2.3% at DR 94–95%, MCC 0.90–0.94 (LycoS17's C5 MCC was 0.93).
  - On nfs18 test: FPR 4.7–5.5%, DR 0.1–1.1%, MCC −0.09 to −0.18. H1 again: a detection collapse with a 2–4×
    rise in FPR.
  - Early stopping kept epoch 1 of 6 in every seed, because validation loss rose afterwards.
- **demo.pcap** (2.5 GB, 272,768 flows; `data/replay/`). Segment A is the held-out 25 minutes of 2017 Wednesday.
  Segment B1 is 2018 09:00–10:25 local: 72 minutes of workstation traffic, then the port-21 "SlowHTTPTest".
  Segment B2 is 13:43–13:45:45 local, including the first ~20 s of Hulk. Each slice is time-shifted to follow the
  previous one by 1 s. Labels are keyed by 5-tuple and start time; 97 keys repeat, and none disagree on the label.
  Seed-0 model at its frozen threshold (`scripts/demo_check.py`, `c13_demo_segments.csv`, `c13_demo_fpr.png`):

| Segment | Flows | Benign | Benign FPR | DR |
|---|---|---|---|---|
| A (2017, held out) | 174,942 | 48,915 | **0.93%** | Hulk 94.3%, SlowHTTPTest 90.4% |
| B1 (2018) | 51,352 | 25,922 | **8.93%** | "SlowHTTPTest" (refused, port 21) 100% |
| B2 (2018) | 46,474 | 1,545 | 7.77% | **Hulk 1.2%** |

  Done-when is met: FPR jumps about 10× on the new network, with 6–10% in each of the first four (all-benign)
  5k-flow windows of B1, against 0.1–1.7% in every window of A. Hulk detection collapses from 94% to 1%.
  Most false positives on B are 2-packet flows to SMB (445), telnet (23), HTTPS (443) and SSH (22). Windows
  hosts and internet scan noise look like the 2017 PortScan family to this model.

## C14: detector API

- **Service:**
  - `src/xnids/live/` holds `api.py` (FastAPI), `service.py` (logic, usable without HTTP), `registry.py` (active,
    candidate, history; promote / reject / rollback; gate checks) and `store.py` (SQLite `live.db` with `scores`,
    `drift` and `actions` as the audit log). The config is `configs/live.yaml`.
  - Endpoints follow the Build Guide, plus `/score/columns` (columnar, for C15 micro-batches), `/models/reject` and
    `/audit`. `/adapt` registers a candidate and never promotes it. Promotion needs `approved_by`, a reason and
    passing gates; refusals are audited.
  - `tests/test_api.py` has 10 tests: 422 on a bad schema, promote needs approval, and rollback restores the
    version.
- **Latency** (`scripts/bench_api.py`, real HTTP to one uvicorn worker, scores stored; `c14_latency.csv`):

| Batch | `/score` p50 / p99 | `/score/columns` p50 / p99 | Columnar throughput |
|---|---|---|---|
| 1 | 6.4 / 8.7 ms | 6.9 / 8.9 ms | 143 flows/s |
| 100 | 12.9 / 21.6 ms | 11.0 / 15.6 ms | 8,900 flows/s |
| 1,000 | 61.7 / 153 ms | 47.4 / 61 ms | 20,900 flows/s |

- **Walkthrough on the demo replay** (`scripts/c14_walkthrough.py`: demo flows → API in 1,000-flow batches, the
  C8 monitor every 5,000 flows → `/drift/report` → `/adapt` → gates → promote → rollback; `c14_walkthrough.csv`):
  - **The C8 monitor does not work on time-ordered real traffic (affects H5).**
    - With the original model it recommends acting in **34 of 35 windows of segment A**, the in-distribution
      held-out 2017 traffic, and misses 8 of 10 B2 windows, where Hulk evades the model.
    - C8's 0 false alarms were measured on *shuffled* windows. Real windows are bursty (one attack, one host's
      activity), and per-feature KS / MMD at n = 5,000 flag every burst.
    - Two causes were measured. First, the reference sample came from the **deduplicated** split, which drops
      the repeated short flows (DNS) that live traffic is full of. Second, even against a raw reference,
      windows of the same network differ by KS 0.5–0.8 from mix alone.
    - Conditioning on predicted-benign flows does not fix it: segment A's KS stays at 0.34–0.89.
    - **To do in C15:** calibrate the monitor on time-ordered, non-deduplicated source traffic (thresholds from
      a null of contiguous windows, not p-values), and act only on an alert that persists for 2 windows.
  - **Adapting inside segment B** (pool = the 10,000 most recent flows, i.e. the first two all-benign B1
    windows; `scripts/c14_actions.sh`, `c14_actions_in_B.csv`):

| Action | Gates | B1 benign FPR afterwards (original 10.75%) | B2 Hulk DR (original 5%) |
|---|---|---|---|
| AdaBN / scaling / CORAL | blocked: canary FPR 41% / 32% / 13% (active 2.2%) | — | — |
| Tent | pass (its guard made no change) | 10.75% | 5.1% |
| MLP few-shot 200 | pass | 13.5% (worse) | 5.7% |
| **XGBoost few-shot 200** | pass | **1.5%** | 4.7% |
| **XGBoost few-shot 1000** | pass | **0.5%** | 1.3% |

  - **Reading:** this is H6 again on live traffic. The label-free adapters fit the new network's statistics and
    then alert on the source's normal traffic; only labelled target flows fix the false positives. The selector,
    restricted to the enabled actions, recommends XGBoost few-shot 200 (predicted gain +0.41). No action recovers
    DoS-Hulk detection on B2, because no Hulk flow from the new network was labelled.
  - **Gate added beyond the Build Guide: canary FPR** (candidate FPR on fixed source-test benign flows ≤ max(2 ×
    active, active + 0.02)). The Build Guide's three gates passed AdaBN, whose benign FPR on B1 went 9% → 56%. The
    attack-rate gate is measured on a buffer that still held segment A's attack-heavy traffic.
  - **Gates also block a bad timing.** Adapting at the first B1 window mixes segment A's Hulk flows into the
    budget. The threshold is then re-picked on Hulk, and the XGBoost candidate fails canary DR (82.7% < 84.4%). The
    same happens when the budget's only attacks are the port-21 refused connections (canary DR 58%).
- **Bug found and fixed in the C9 few-shot adapter.**
  - With fewer than `min_attacks` attacks in the budget, it kept the frozen source threshold even though the model
    had been retrained. A retrained XGBoost then alerted on **every** flow (threshold 0.0012, minimum score
    0.0044). It now re-picks on source validation, as CORAL/DANN do (test added).
  - C10 v2 rows that used the old fallback, estimated from each window's attack share: about 6 of 800 XGBoost
    few-shot rows and about 58 of 400 MLP few-shot-50 rows. The C11 conclusion (best fixed action = XGBoost
    few-shot 200) does not rest on them, and C10 / C11 have not been re-run.

## C15: replay runner and the calibrated live monitor

- **The monitor was re-calibrated for time-ordered real traffic** (`scripts/calibrate_monitor.py`,
  `xnids.drift.novelty`, `DriftMonitor` mode `calibrated`; C8's default mode is unchanged).
  - **Reference:** 20,000 **raw**, non-deduplicated 2017 flows. **Null:** the other 1.34M raw flows in time order
    (268 windows of 5,000), source attack bursts included. The demo's segment A (the split holdout) is in neither.
  - **No KS / MMD threshold separates the networks.** Source attack bursts reach `ks_max` 0.90 and `mmd_stat`
    0.64, so at the null's 99th percentile 0% of segment-B2 windows exceed any of them. A change of **mix** and a
    change of **network** look alike to distribution-shape tests.
  - **New statistic: the novelty share.** It is the share of a window's flows whose nearest reference flow, in the
    MLP's standardised penultimate-layer embedding, is farther away than 99% of held-out source flows are. The
    window threshold is the null's 99th percentile, 3.2%.
    - Windows over the threshold: segment A 2.9%, B1 100%, B2 80%.
    - In raw input space, B2 reached only 20%: the evasive Hulk flows look ordinary in raw features.
  - **Alert = 2 consecutive windows over the threshold.** The ATC cost trigger is dropped in this mode because it
    estimated a larger error rise for segment A (0.04–0.10) than for B1 (0.03–0.07).
  - **Result on the demo stream:** **0 of 35 segment-A windows alert**, and the first alert fires at the 2nd
    segment-B window (10,000 flows after the switch).
  - **H5 caveat:** C8's 0 false alarms hold for shuffled windows only. The C8 detector flagged 34 of 35
    in-distribution windows of real time-ordered traffic.
- **Runner** (`xnids.live.replay`, `scripts/replay.py`):
  - File mode runs NFStream on the PCAP, live mode runs NFStream on an interface, and parquet mode reads
    pre-extracted flows (for the load test).
  - Every 200 ms it sends what is available, at most 1,000 flows per request, to `/score/columns`.
  - The monitor processes each 5,000-flow window in its own thread and posts the report to `/drift/report`.
  - Optional rehearsal policy: auto-adapt on the first alert, auto-approve when the gates pass.
  - **Bug found and fixed: NFStream with 4 meters emits flows out of order.** Each meter processes its share of
    the packets at its own pace, so segment-B flows filled 16–50% of the windows that should hold segment A.
    The median emission lag was 4,066 s of capture time. The runner now uses one meter: windows are 94–100% one
    segment and the median lag is 0.1 s. The meter count does not change a flow's content, so the frozen
    extractor hash still applies.
  - **Emission lag** (time between a flow's last packet and its report, in capture time): median 0.1 s, p95 482
    s. Flows that end with a FIN/RST are reported at once; others wait out the 120 s idle timeout, and long flows
    wait longer.
- **Whole demo.pcap, file mode, unattended rehearsal** (`data/live/runs/file-rehearsal`, `c15_rehearsal.png`):
  - 272,768 flows in 58 s, about 4,700 flows/s end to end; `/score` p99 60 ms.
  - Segment A: 0 alerts, benign FPR 0.1–3%.
  - Alert at window 36. The selector recommends XGBoost few-shot 200, the gates pass and the candidate is promoted.
  - Segment B1 benign FPR: 10.9% and 8.3% before, **0.06% and 0.2%** afterwards (windows 38–39). Window 37 was
    still scored by the old model, because the monitor thread runs one window behind scoring.
  - **Cost of adapting on benign-only new-network flows:** the promoted model no longer flags B1's port-21 refused
    connections (DR 0%; the original MLP flagged 100%), and B2 Hulk stays missed.
- **Load test** (`scripts/load_test.py`, `c15_load.csv`, `c15_load.png`): pre-extracted flows, monitor on, one API
  worker, scores stored.

| Target rate (flows/s) | Achieved | p50 / p99 (ms) |
|---|---|---|
| 1,000 | 1,000 | 16.5 / 31.2 |
| 2,000 | 2,000 | 23.6 / 40.5 |
| 5,000 | 5,000 | 49.3 / 74.2 |
| 10,000 | 10,000 | 49.6 / 72.7 |
| top speed | 16,880 | 44.8 / 72.4 |

  **The sustained rate with p99 < 100 ms is about 17k flows/s.** End to end, NFStream on one meter is the
  bottleneck, at about 4,700 flows/s.
- **Live mode** (`scripts/replay_live.sh`) needs root: a veth pair, tcpreplay and NFStream on `veth1`.
  - Timing features only match file mode at multiplier 1, because `--multiplier` compresses packet gaps. The
    file-vs-live parity check (`scripts/replay_parity.py`) therefore uses a 10-minute slice at x1, A → B1:
    73,494 flows in file mode.
  - Live flows carry wall-clock start times, so the evaluation overlay falls back to the 5-tuple where it carries
    one label (96.5% of demo flows).
  - **Live mode has not been run yet:** it needs the user's sudo.

## C16: dashboard and the 5-minute demo

- **Dashboard** (`dashboard/app.py`, Streamlit; `make demo` = a fresh API plus the dashboard, bound to localhost).
  Four pages that refresh every 2 s:
  - **Live:** flows/s, alerts/min, p99 latency, and benign FPR / DR per 5,000 flows for the *original* vs the
    *adapted* model. This is the evaluation overlay from the replay labels.
  - **Drift:** novelty per window against the calibrated threshold, top shifted features (KS), MMD p-value and
    ADWIN marks, and the plain-language explanation.
  - **Adapt:** the selector's recommendation with predicted gain ± std, Build candidate, the gate table, Approve /
    Reject (name and reason required), Rollback, and Reject & rebuild.
  - **Audit.**
- **Supporting changes:**
  - The API now keeps scoring the first deployed model as an **"original" shadow** after a promotion, so original
    vs adapted can be compared on the same traffic.
  - The runner takes a **rate schedule**: 2,000 flows/s through segment A, so the switch falls at about 1:30, then
    300 flows/s in segment B.
  - The runner writes a status file (rate, flow-expiry lag) for the sidebar.
- **Rehearsals** (`scripts/rehearse_demo.py`: the real app.py under Streamlit AppTest, a fresh API per run, the
  presenter's clicks scripted; `c16_rehearsals.csv`). **3 of 3 ran start to finish.** Times are from Start:

| Run | Switch to B | Drift alert | Candidate | Approved | FPR 40 s later: original vs adapted | Rebuilds |
|---|---|---|---|---|---|---|
| 1 | 91.8 s | 125.7 s | 127.0 s | 127.9 s | 7.4% vs 0.4% | 1 (gate blocked canary FPR) |
| 2 | 91.4 s | 125.3 s | 127.0 s | 127.2 s | 7.4% vs 0.8% | 0 |
| 3 | 93.2 s | 124.9 s | 126.3 s | 126.6 s | 7.4% vs 0.7% | 0 |

  In every run, segment A raised no alert. Rollback restored the original and the audit trail was in order.
- **Found by rehearsing: the recommended action is occasionally unstable.**
  - In an earlier 3× rehearsal, run 3's XGBoost few-shot-200 candidate failed the canary-FPR gate (10.9% vs a limit
    of 4.4%).
  - Training is deterministic, including with concurrent GPU scoring. The cause is the pool: the 10,000 most recent
    flows when the analyst clicks.
  - Over 41 click timings, budget 200 at target weight 0.2 (the C9 default) fails the gates 1 time in 41. Target
    weight 0.1, or budget 1,000, fails 0 of 41.
  - The adapter was **not** re-tuned on the demo stream, since that would be tuning on test traffic and would change
    the C9–C11 results. The gate blocks the bad candidate, and the dashboard offers **Reject & rebuild with a new
    label sample** (next seed, audited). The rehearsal takes that branch when the gates fail.
- **Not verified here:** the visual layout in a real browser (headless Firefox captures before Streamlit renders).
  No backup screen recording yet; both are for the presenter (`docs/DEMO.md`).
