# DriftGuard: results by hypothesis

A technical results digest for the report and the paper. Every number below comes from a table or figure in
`reports/`. `reports/PROVENANCE.csv` maps each file to the command that produced it, the MLflow experiment, and
(`reports/provenance/<step>_runs.csv`) the run IDs behind it. `python scripts/make_tables.py` rebuilds every
MLflow-backed table and figure from those runs (byte-identical on the last rebuild: 152 of 152 files). The detailed,
chronological record, including every bug found and fixed, is `reports/FINDINGS.md`. Data: `DATASET_CARD.md`.
Models: `reports/model_cards/`.

## Summary

| | Hypothesis | Verdict | Key evidence |
|---|---|---|---|
| H1 | Every model does worse cross-dataset than within | **Supported, as a detection and ranking collapse** | 134 of 136 cross cells lose MCC and oracle FPR; frozen-threshold FPR rises in only 91 |
| H2 | Original labels inflate scores | **Not supported once the split is controlled** | ≤ 0.008 MCC difference on like-for-like splits; the split scheme matters more (0.03–0.08) |
| H3 | Removing / normalising lab-telling features explains part of the collapse | **Not supported: ~0% explained** | median share 0.000 over 96 cells; per-domain normalisation is worse than nothing |
| H4 | Gradual drift raises false positives less than a sudden switch | **Supported** | autoencoder benign FPR 1.3% on later days vs 58.3% on a new network |
| H5 | The detector flags a switch faster / cheaper than scheduled retraining | **Supported offline; qualified on real traffic** | 0 false alarms and 5,000-flow delay on shuffled windows; the C8 rule false-alarms on time-ordered traffic, fixed by the C15 calibrated novelty monitor |
| H6 | Tent / AdaBN / meta-learning recover more than CORAL / DANN, at lower label cost | **Partly supported, mostly not** | label-free methods help only within one extractor family; labels dominate; Reptile = fine-tuning |
| H7 | An unguarded self-updating adapter can be poisoned; safeguards reduce it | **Supported (with an adaptive attacker)** | Tent loop: DDoS DR 100% → 1.6%; anchored canary / rollback keep 98–100% |
| H8 | Re-extracting the same PCAPs with another tool changes performance | **Supported** | same 2018 PCAPs, two extractors: RF / XGB / MLP MCC 0.96 → 0.02–0.18 |

## Data and method in one paragraph

Six datasets are used: LycoS-IDS2017 and LycoS-Unicas-IDS2018 (the label-corrected LycoSTand re-extractions of
CIC-IDS2017 / CSE-CIC-IDS2018), NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2, and the original CIC-IDS2017
CSVs (H2 only). They are compared on four feature tracks: cic77, nf43, a 15-feature core map shared by both
schemas, and cic_orig. Rows are deduplicated before splitting (NetFlow-v2 loses 80–90% as exact duplicates).
LycoS17 is split in 5-minute time blocks, the others stratified. Every split is frozen and hashed. Seven models
(LDA, DT, RF, XGB, MLP, autoencoder, TabNet) are trained with 3 seeds. **The decision threshold is fixed at 95%
detection on source validation and never tuned on the target.** The headline metric is FPR at that threshold,
always reported with DR, plus PR-AUC, MCC and oracle FPR@95%DR as diagnostics. For the dataset details see
`DATASET_CARD.md`.

## H1: collapse across datasets

Evidence: `tables/matrix_{cic77,nf43,core}.csv`, `h1_checks.csv`, `figures/matrix_*`, `within_vs_cross_*`
(step `c6_matrix`, 207 runs).

| Track | Within MCC | Cross MCC | Within DR | Cross DR | Within oracle FPR@95%DR | Cross oracle FPR@95%DR |
|---|---|---|---|---|---|---|
| cic77 | 0.910 | 0.161 | 0.909 | 0.264 | 0.022 | 0.646 |
| nf43 | 0.874 | 0.086 | 0.951 | 0.357 | 0.056 | 0.923 |
| core | 0.864 | 0.120 | 0.928 | 0.271 | 0.099 | 0.699 |

- **134 of 136 cross cells are worse** in MCC and oracle FPR. The two exceptions are LDA on the bridge pair (the
  same traffic, two extractors; see H8).
- **The collapse is mostly silent.** At the frozen threshold, FPR rises in only 91 of 136 cells: supervised models
  flag almost nothing on a new network (random forest's median cross DR is 0.01%). The autoencoder fails the other
  way, flagging 92–100% of benign flows on every nf43 cross pair. H1 should therefore be stated as a detection and
  ranking collapse, and FPR is always reported with DR.
- **Transfer is asymmetric and favours diverse training data.** LycoS18 → LycoS17 reaches MCC 0.41–0.57 for five
  models, while LycoS17 → anything is ≤ 0.014. RF trained on NF-ToN reaches 0.851 on LycoS18 across extractor
  families.
- **Cantone et al. (2024) reproduced** (`cantone_comparison.csv`): within-dataset MCC 0.933 vs their 0.946; LDA
  LycoS18 → LycoS17 0.559 vs 0.604. But "LDA generalises best" does not hold on average: on cic77 the MLP leads
  (cross MCC 0.28), and on nf43 LDA is near the bottom (0.008).

## H2: label inflation

Evidence: `tables/h2_original_vs_corrected.csv`, `matrix_h2_control_*` (step `c6_matrix`).

- **Compared like for like** (the corrected release re-split at random, as the original), the original CIC-IDS2017
  is at most 0.003–0.008 MCC higher for DT / RF / XGB / MLP / TabNet. PR-AUC differs by ≤ 0.0007.
- For LDA (−0.22) and the autoencoder (−0.08) the original release is **worse**.
- **The split scheme matters more than the release.** Time blocks alone cost the corrected release 0.03–0.08 MCC;
  an uncontrolled comparison would have mistaken this for a large H2 effect.
- **Confound:** the releases also differ in extractor (CICFlowMeter vs LycoSTand), so this compares releases, not
  label errors in isolation.

## H3: what explains the collapse

Evidence: `tables/advval_summary.csv`, `advval_lab_features.json`, `figures/advval_*` (C4, 84 runs);
`tables/c7_shares.csv`, `c7_norm_shares.csv`, `figures/c7_share_mcc.png` (C7, 333 runs).

- **Datasets are trivially separable** (adversarial validation), and they stay separable after removing the 10
  most domain-identifying features. The "lab signature" is spread across the whole feature space.
- **Removing the top-k lab-telling features** (k = 1, 3, 5, 10; RF / XGB / MLP; 8 pairs × 3 seeds) explains a
  median **0.000** of the collapse (IQR −0.007 to 0.007). It costs nothing within-dataset either.
- **Per-domain normalisation is worse than doing nothing**: mean cross MCC falls from 0.104 to −0.003, erasing
  the partial transfer that existed (LycoS18 → LycoS17 MLP 0.572 → 0.102).
- The Build Guide's frozen-threshold FPR share formula has a non-positive denominator in 48 of 96 cells (silent
  failure), so MCC, oracle-FPR and PR-AUC shares are used.

## H4: gradual vs sudden drift

Evidence: `tables/h4_summary.csv`, `h4_by_day.csv`, `figures/h4_benign_fpr_over_time.png` (step `c7_h4`).

- Models are trained on LycoS17 Monday–Tuesday and evaluated hour by hour over Wednesday–Friday, then on
  LycoS18.
- The benign-only autoencoder's FPR stays **1.25–1.38%** on later days (feature KS 0.057 → 0.100). On the new
  network it is **58.3%** (KS 0.391).
- **Confound:** each CIC-2017 day runs a different attack, so benign-only measures are the primary evidence. The
  supervised models stay silent on every later day and on LycoS18 (FPR ≤ 0.04%), so their FPR cannot show drift.

## H5: detecting the switch

Evidence: `tables/drift_h5.csv`, `drift_vs_retraining.csv`, `drift_explanations.csv`, `figures/drift_switch_*`
(C8, 24 runs); `tables/c15_monitor_{null,demo}.csv`, `figures/c15_rehearsal.png`, `model_cards/drift_monitor.md`
(C15).

- **Offline, on shuffled windows** (8 pairs × 3 seeds), every detector catches a network switch within the first
  5,000-flow window. The combined rule raises 0 false alarms per 100 null windows.
- **Cost:** under the stated cost assumptions, one switch per day costs 2,000 units with the monitor, against
  4,800 (with 100× more unhandled flows) for retraining every 1M flows. The ATC cost trigger misses a third of
  switches, because models fail silently and stay confident.
- **Explanations agree with C4:** on nf43, all 5 top drifted features after a switch are C4 lab-telling features.
- **Qualification on real, time-ordered traffic (C14/C15).** The same rule false-alarms on 34 of 35 windows of
  in-distribution 2017 traffic. Attack bursts and host activity look like drift to KS / MMD, and a deduplicated
  reference lacks the repeated short flows live traffic is full of.
- **The fix (C15)** is the share of flows unlike any raw reference flow (nearest neighbour in the MLP's
  embedding), thresholded at the 99th percentile of 268 time-ordered 2017 windows, plus a 2-window persistence
  rule. It gives **0 of 35** alerts on in-distribution windows, and the first alert at the 2nd window of the new
  network.

## H6: recovering from drift

Evidence: `tables/c9_summary.csv`, `figures/c9_before_after.png` (C9); `c10_summary.csv` (C10);
`c11_summary.csv`, `figures/c11_regret.png` (C11); `c12_summary.csv`, `c12_paired.csv`, `figures/c12_h6.png` (C12);
`c14_actions_in_B.csv` (live).

- **Label-free adaptation helps only within one extractor family.**
  - On LycoS17 ↔ LycoS18, AdaBN roughly doubles MLP MCC (0.28 → 0.54) for +1.2 FPR points; Tent, CORAL and DANN
    follow.
  - On every NetFlow pair, every label-free adapter makes things worse.
- **Labels dominate.** XGBoost retrained with 1,000 labelled target flows reaches MCC 0.915 (cic77) and 0.940
  (nf43). In C10's full-information logs a labelled action is best in 98–99% of windows at zero label cost.
- **Learned selection does not beat the best fixed action.** Leave-one-pair-out regret is 0.077 for the selector
  vs 0.050 for always choosing XGBoost few-shot with 200 labels. It wins on 3 of 8 pairs and offers waiting and
  uncertainty.
- **Meta-learning adds nothing.** Reptile equals fine-tuning from the same start with the same labels (core:
  paired difference −0.004 to +0.013). It is worse on nf43 (−0.06 to −0.10).
- **Live confirmation:** on the demo's new network, AdaBN / scaling / CORAL candidates are blocked by the gates
  (canary FPR 13–41%). XGBoost few-shot 200 / 1000 cuts segment-B1 benign FPR from 10.75% to **1.5% / 0.5%**.

## H7: poisoning the adaptation loop

Evidence: `tables/c17_summary.csv`, `c17_rounds.csv`, `figures/c17_{ddos,dos}_{tent,adabn}.png` (step
`c17_poison`).

- **Setup:** the demo MLP adapts itself for 30 rounds on its own network. An attacker controls 1–10% of each
  round's traffic and cannot touch labels or the model. 3 seeds, paired pools.
- **The Build Guide's frog-boiling and statistic-skew attacks degrade nothing**, and continual AdaBN resists
  every attack.
- **An adaptive attacker who can query the detector's score breaks an unguarded Tent loop.** DDoS detection holds
  for about 18 rounds, then collapses to **1.6% ± 2.6** at 10% injection (65% at 5%); DoS falls 84% → 67%.
- **The attack is stealthy:** benign FPR stays at 0.9% (0.7% clean) and the other families barely move.
- **Safeguards anchored to the original model work:** canary (DDoS 98%, DoS 83%), rollback (100% / 83%), all
  guards (98% / 83%).
- **Per-step safeguards do not:** the C14 gate (89% / 67%) and a per-step canary (95% / 67%) never fire on the
  slow DoS drift. An L2 update clip never binds.
- **Limitation:** a family absent from the canary set would likely evade every canary-based guard.

## H8 (stretch): extractor sensitivity

Evidence: `tables/bridge_check.csv`, `bridge_check_summary.csv` (C2); `tables/matrix_core.csv` (C6);
DATASET_CARD C13 section.

- **Same PCAPs, two extractors:** LycoS18 (LycoSTand) and NF-CSE-CIC-IDS2018-v2 (NetFlow-v2) both come from the
  CSE-CIC-IDS2018 PCAPs.
  - On benign traffic, 15 of 18 mapped core features agree once units match.
  - Within attack families the flows differ wildly (median per-family KS of packet / byte counts 0.78–0.98; 1.80M
    vs 0.43M DoS-Hulk flows).
- **Performance on the core track:** RF / XGB / MLP reach MCC 0.96–0.97 within either extraction, and **0.02–0.18
  across them** (DR 0.1–4%). That is as bad as moving to a different network. Only LDA transfers (0.83 / 0.55).
- **A third extractor (C13):** NFStream needed a TCP-termination plugin before its flow counts matched LycoSTand's.
  Without it, about 12 DoS-Hulk connections merged into one flow.

## Live system (C13–C16)

Evidence: `tables/c13_demo_segments.csv`, `c14_latency.csv`, `c14_walkthrough.csv`, `c15_load.csv`,
`c16_rehearsals.csv`; `figures/c13_demo_fpr.png`, `c15_load.png`, `c15_rehearsal.png`; `docs/DEMO.md`.

- **Demo data:** CIC-2017 and CSE-CIC-IDS2018 PCAPs → frozen NFStream extractor → API.
- **Performance:** about 17k flows/s with p99 < 100 ms (one worker); end to end about 4,700 flows/s (single-meter
  NFStream is the bottleneck).
- **Demo replay:** on the new network the demo model's FPR jumps from 0.93% to 8.9%, and its Hulk detection falls
  from 94% to 1.2%.
- **Five-minute demo, rehearsed 3 of 3 times through the real dashboard:** the monitor alerts about 33 s after the
  switch. The selector recommends XGBoost few-shot 200, the gates pass and a human approves. The adapted model's
  FPR is 0.4–0.8% vs the original's 7.4% on the same traffic. Rollback and the audit trail work.

## Limitations

- **Data:**
  - Laptop dev subsample (≤ 2M rows per split) for C5–C12.
  - Public lab datasets only.
  - NetFlow-v2 and LycoS18 have no timestamps (stratified splits).
  - The 2018 "DoS-SlowHTTPTest" in the demo is refused port-21 connections.
- **Seeds:** LDA and XGBoost train deterministically, so their 3 seeds coincide (model cards).
- **Thresholds:** fixed at 95% source DR by design; per-deployment recalibration is not studied.
- **H5 live monitor:** calibrated on one source network; its threshold must be recalibrated for any other.
- **H6:** selector and Reptile evaluated on 8 pairs; the label oracle in the live system is the replay ground
  truth.
- **H7:** offline, one network, MLP only, 30 rounds; the adaptive attacker needs score queries.
- **Live mode** (tcpreplay into a veth pair) needs root and has not yet been compared with file mode
  (`scripts/replay_parity.py`).
- **Visual check:** the dashboard has not been checked in a browser by the authors of this digest.

## Chapters still to write (team)

Introduction, Literature review and Conclusion are deliberately not drafted here. Build Guide C18 asks for each
member to be able to explain the pipeline (ownership), and the paper's framing and literature positioning are
the team's call. Suggested core story (Build Guide): cross-schema measurement + decomposition + learned adapter
choice + poisoning of the adaptation loop, now with the two strongest additions: extractor sensitivity (H8) and
the time-ordered monitoring correction (H5).
