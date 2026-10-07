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
