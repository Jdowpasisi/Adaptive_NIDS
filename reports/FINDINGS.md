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
