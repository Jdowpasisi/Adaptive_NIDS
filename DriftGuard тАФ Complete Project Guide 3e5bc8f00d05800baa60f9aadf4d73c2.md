# DriftGuard — Complete Project Guide

*Cross-Dataset Generalisation and Drift-Aware Adaptation for ML-Based Network Intrusion Detection*
B.Tech/M.Tech CSE (Cyber Security) · VII Semester Project-1 · National Forensic Sciences University, Dharwad Campus

**Purpose of this document:** this compresses every planning artefact the team has produced — the TA1 slide content, both individual documentation drafts, the technical guide, the full literature review, and the fine-tuning design discussion — plus a full breakdown of the peer-reviewed paper that anchors our Phase 1, into one place. Read this top to bottom once, then use it as a reference. If any other document disagrees with this one, this one wins — it's the most recent synthesis (25 September 2026).

---

## Part 1 — The Big Picture

### 1.1 The elevator pitch

Machine-learning intrusion detectors report 99%+ accuracy on lab benchmarks, yet SOCs (security operations centres) still don't trust them. We're measuring exactly how much of that 99% survives being pointed at a *different* network, on data with corrected labels so the numbers are honest. We then explain where the loss comes from, build a system that detects it happening in production and explains it in plain language, and test whether the model can recover the loss cheaply — using adaptation methods and, as the newest piece, a fine-tuning system that learns *how* to adapt from data rather than us hand-tuning it. We package the whole thing as a working pipeline that replays real traffic and shows before/after side by side, and we stress-test it with a controlled poisoning attack.

**We are not inventing a new detection algorithm.** The contribution is a rigorous measurement, a working explain-and-adapt system, and (as of this update) a principled, non-RL fine-tuning layer on top of it.

### 1.2 Why this project exists

| Cause | Evidence | Our response |
| --- | --- | --- |
| Lab traffic isn't real traffic — ML-NIDS operate in an "open world" where mistakes are costly | Sommer & Paxson, 2010 | Test explicitly across different networks |
| Labels contain errors — flows are wrongly built and mislabelled | Engelen et al., 2021; Liu et al., 2022 | Use label-corrected datasets |
| Evaluation flatters the model — data snooping, spurious correlations, lab-only testing | Arp et al., 2022 | Leakage-free, time-based splits |
| Traffic drifts after deployment — new traffic and attacker tooling move away from training data | Yang et al. (CADE), 2021 | Drift alarm + domain adaptation |

**Motivation, by audience:**

- **SOC teams:** false positives cause alert fatigue; a model that doesn't transfer has to be retrained for every network it's deployed on.
- **Researchers:** the field needs honest, reproducible cross-network results instead of single-dataset scores that don't hold up.
- **Indian organisations specifically:** very little labelled local network traffic exists publicly, so the practical route is transferring models trained on public (mostly Western) benchmarks — which is exactly the transfer this project measures and tries to fix.

### 1.3 Problem statement

> ML-based network intrusion detection models trained on one benchmark dataset degrade on traffic from a different network, and their within-dataset scores are inflated by known labelling errors. This project measures that degradation on harmonised, label-corrected datasets and develops a drift-aware pipeline that detects distribution shift and adapts to it, evaluated using false-positive rate, detection rate, and detection latency.
> 

---

## Part 2 — Phase 1: The Paper We're Building On

Everyone on the team must be able to explain this paper in the viva. It is the closest published work to our Phase 1/2 (the collapse-matrix measurement), it independently confirms our core hypothesis with real numbers, and it also tells us exactly where published work *stops* — which is where our project begins.

**Cantone, Marrocco & Bria (2024), "Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study," IEEE Access, vol. 12, pp. 144489–144508.** Open access (CC BY 4.0), published 3 October 2024, University of Cassino and Southern Latium, Italy.

### 2.1 What they actually did

They trained four classical ML classifiers on four datasets and tested every model on every dataset (within-dataset and cross-dataset), to answer one question: **does an NIDS model trained on one network still work on a different one?**

**Datasets (4):**

| Dataset | Origin | Role |
| --- | --- | --- |
| CIC-IDS2017 | Canadian Institute for Cybersecurity (CIC), 2017, 14 machines | Original benchmark |
| CSE-CIC-IDS2018 | CIC + Communication Security Establishment, 2018, 500 AWS machines | Original benchmark, larger scale |
| **LycoS-IDS2017** | Rosay et al. — a *corrected* re-extraction of CIC-IDS2017's raw PCAPs using a new tool, LycoStand, fixing feature duplication, miscalculation, wrong protocol detection, and labelling errors | Label-corrected version of CIC17 |
| **LycoS-Unicas-IDS2018** | **New in this paper** — the authors applied LycoStand to CSE-CIC-IDS2018's raw PCAPs themselves, producing a corrected 2018 counterpart. Publicly released at `github.com/MarcoCantone/LycoS-Unicas-IDS2018` | Label-corrected version of CIC18 |

**This is a genuinely useful discovery for us:** we no longer need to apply Liu et al.'s correction ourselves — LycoS-IDS2017 and LycoS-Unicas-IDS2018 are both already public, already built with the *same* extraction tool (LycoStand), which means they're a clean, consistent, already-corrected CIC17/CIC18 pair. **Recommendation: adopt these two directly as our label-corrected CIC-family datasets**, rather than re-deriving corrections from Engelen/Liu's papers by hand.

**Models (4):** LDA (Linear Discriminant Analysis), Decision Tree, Random Forest, XGBoost. Note: no neural models — this is a purely classical-ML study, which is one of the gaps we fill (we add MLP, an autoencoder, and TabNet).

**Method:**

- Binary classification (all attacks grouped as one "malicious" label) for the main experiment, plus separate **single-attack** experiments (one attack class vs. benign at a time).
- MinMax normalisation; grid search on 20% of training data (search spaces in their Table 3); retrain on full data with best hyperparameters.
- **mRMR (minimum Redundancy Maximum Relevance)** feature selection — chosen over dimensionality reduction (like PCA) specifically to keep features interpretable, then swept the number of retained features (1, 2, 3, 4, 5, 10, 20) to see how performance scales.
- Visual analysis: KDE (kernel density estimation) and PCA plots of the top-2 mRMR features per attack, comparing how the *same* attack class is distributed in each dataset.
- Metric: **MCC (Matthews Correlation Coefficient)**, plus F1 and AUROC — explicitly **not accuracy**, because 11 of 14 attacks in CIC17 have a benign:attack ratio over 200:1, so a model that always predicts "benign" would score 99.5% accuracy while being useless. (This is the same reasoning behind our own choice of FPR@95%DR over accuracy — good independent confirmation.)

### 2.2 Their headline findings (know these numbers)

- **Average MCC: 94.63% within-dataset vs. 29.35% cross-dataset**, averaged across all four classifiers and all train/test combinations. None of the four classifiers transferred learned patterns to a new network.
- Worst cross-dataset pair: **LycoS17 → LycoS18, MCC 10.83%.** Best cross-dataset pair: **CIC18 → CIC17, MCC 40.24%.**
- Training on the larger, more varied CSE-CIC-IDS2018 (13.7M instances) gave *worse* within-dataset scores but *better* cross-dataset transfer than training on the smaller CIC-IDS2017 (1.8M instances) — evidence that a model that fits its training set too perfectly is a red flag for generalisation, not a good sign.
- **LDA — the simplest model — got the single best cross-dataset result of the whole study** (60.41% MCC, LycoS18→LycoS17), better than RF or XGBoost in that direction. Their read: lower model complexity reduced overfitting to lab-specific quirks. This is directly relevant to our own model line-up: don't assume the fanciest model (TabNet, autoencoder) will generalise best — measure it.
- **Single-attack experiments:** no classifier generalised a specific attack's pattern across every train/test combination. 7 attacks (DoS GoldenEye, Slowloris, Hulk, Slowhttptest, DDoS LOIC-HTTP, SSH-Patator, FTP-Patator) transferred *partially* in about a third of configurations (MCC > 50%); the 3 web attacks and Infiltration essentially never transferred (too few samples to learn a real pattern in the first place).
- **Feature-count experiments:** using only 3–5 mRMR-selected features often matched or beat using the full ~77-feature set for **within-dataset** performance — but cross-dataset performance stayed low regardless of feature count. Their interpretation, which matters a lot for our own decomposition step (H3): attacks in a single dataset are so *homogeneous* that a handful of features can perfectly separate them within that dataset, without those features capturing anything about the attack that would hold true in a different network.
- **The visual analysis is the paper's most important finding for us:** plotting the same attack class's samples from two different datasets in the same 2D feature subspace showed **substantially different distributions for 10 of 11 attacks** (DoS Slowloris was the one exception). In several cases (DDoS LOIC-HTTP, DoS Slowhttptest) one dataset's version of the attack was a single repeated point/tiny cluster — meaning the "attack" in that dataset is really just one specific tool run with fixed parameters, not a representative sample of that attack in general. This is direct visual evidence for our **H1 (collapse)** and strong supporting evidence for **H3 (decomposition)** — much of the cross-dataset failure isn't "the model is bad," it's "the datasets don't actually represent the same thing," which is exactly what our adversarial-validation decomposition step is designed to quantify.

### 2.3 Their own conclusion, and — importantly — what they explicitly did NOT do

Their conclusion: a model trained on a single public dataset is not a robust NIDS, because these datasets don't adequately represent the *real* distribution of attacks — they represent one specific execution of that attack, with that lab's specific tools and parameters. They explicitly note this "casts doubt on the practicality of employing ML for designing NIDS" without further work — which is exactly the doubt our project is answering.

**What they stopped short of (this is our gap, precisely):**

- No drift detection — they only measured static cross-dataset gaps, never a live/streaming scenario.
- No domain adaptation attempt of any kind — they diagnose the problem but never try to fix it.
- No NetFlow-schema datasets — they explicitly excluded NetFlow datasets from their study because the classes don't overlap with CIC's attack taxonomy. This means our NF-UNSW-NB15-v2 / NF-ToN-IoT-v2 cross-lab, cross-schema experiments are **genuinely untouched territory relative to this paper.**
- No live/systems component — no PCAP replay, no served model, no dashboard, no latency measurement.
- Their own suggested future work: "more powerful deep learning models that learn features directly from raw PCAP data" and federated learning with properly held-out test networks — neither of which is quite our direction, but confirms that classical ML + a real fix (not just diagnosis) is still an open space.

**Strategic implication for our plan:** Cantone et al. have already published almost exactly our planned Phase B (within- vs. cross-dataset collapse matrix) on the CIC/LycoS side, with real numbers, in a peer-reviewed IEEE Access paper. We should:

1. **Cite it as direct confirmatory evidence** for H1 rather than treating our own CIC/LycoS collapse-matrix run as a novel result — we can run it faster/lighter (e.g., fewer seeds, or skip it if time is tight) since the core finding is already established in the literature.
2. **Reallocate the time saved toward what they didn't do** — drift detection, adaptation, meta-learning-based fine-tuning, and the NetFlow cross-lab experiments — since that's where our actual contribution now clearly sits.
3. When asked "isn't this already done?" in the viva, the answer is: *"Cantone et al. (2024) proved the collapse is real on CIC-family data with corrected labels — we build the explain-detect-adapt system on top of that confirmed finding, and extend the measurement itself to cross-schema NetFlow datasets they explicitly excluded."*

---

## Part 3 — What We're Building on Top (Phase 2 onward)

| Capability | Cantone et al. (2024) | DriftGuard (this project) |
| --- | --- | --- |
| Cross-dataset measurement | Yes (CIC-family only) | Yes — extended to NetFlow-schema, cross-lab datasets |
| Label-corrected data | Yes (LycoS) | Yes — reusing LycoS + LycoS-Unicas directly |
| Explains *why* it collapses | Partial (KDE/PCA visualisation only) | Yes — adversarial validation decomposition, quantified |
| Drift detection (live/streaming) | No | Yes (KS, MMD, ADWIN) |
| Domain adaptation | No | Yes (CORAL, DANN, AdaBN, Tent, few-shot) |
| Learned fine-tuning strategy | No | **Yes — supervised reward model + meta-learning (new)** |
| Live pipeline / systems demo | No | Yes (PCAP replay → API → dashboard) |
| Adversarial robustness of the fix itself | No | Yes — poisoning test on the adaptation loop |

---

## Part 4 — System Architecture

This is the actual architecture diagram from our TA1 materials — three lanes: **Offline** (build the model), **Online** (serve it live), **Control** (watch it and fix it when it drifts).

![DriftGuard system architecture](https://app.notion.comassets/architecture_diagram.jpg)

**Walking through it:**

1. **Offline lane:** public datasets → harmonised and split into a shared schema → RF/XGBoost/MLP (and TabNet) trained and evaluated, tracked in MLflow → benchmark results (the collapse matrix).
2. **Online lane:** a PCAP is replayed with `tcpreplay` at controlled speed → flows are extracted live with **the same feature schema** as offline training (this consistency is critical — see Risk table in Part 10) → the Detector API (FastAPI, serving the deployed model) scores each flow → results reach a SOC-style dashboard showing alerts, FPR, and drift status.
3. **Control lane:** the drift monitor (KS/MMD/ADWIN) watches the live flow features → when drift is flagged, the adapter (CORAL/DANN, plus our AdaBN/Tent/meta-learning extensions) produces a **candidate** updated model → a human approval gate reviews it before it's deployed back into the Detector API. This human-in-the-loop step is also our main defence against the poisoning scenario (Part 9).

---

## Part 5 — Objectives and Hypotheses

### 5.1 Objectives

| # | Objective | Measure of success | Due |
| --- | --- | --- | --- |
| O1 | Design a harmonised, leakage-free benchmark from ≥3 public IDS datasets with corrected labels | Dataset card, preprocessing scripts, fixed splits | Mid-Sem |
| O2 | Evaluate within- vs. cross-dataset performance of RF, XGBoost, MLP, with original vs. corrected labels | N×N matrix of FPR@95%DR and PR-AUC | Mid-Sem |
| O3 | Implement a drift detector for incoming network flows | Detection delay, false-alarm rate | TA2 |
| O4 | Compare adaptation methods (Deep CORAL, DANN, + AdaBN/Tent/meta-learning) for reducing cross-dataset false positives | FPR@95%DR before vs. after | TA2 |
| O5 | Develop a live PCAP-replay prototype with an alert dashboard | Per-flow latency (ms), flows/second | TA2 / End-Sem |
| O6 | Prepare a research paper for submission after supervisor approval | Manuscript + submission evidence | End-Sem |

### 5.2 Hypotheses (plain language, with fallback if wrong)

| # | Hypothesis | Plain meaning | If it fails |
| --- | --- | --- | --- |
| H1 | Collapse | Every model does worse cross-dataset than within-dataset — **already confirmed independently by Cantone et al. 2024** | N/A — treat as established, cite it |
| H2 | Label inflation | Scores on original (uncorrected) labels are higher than on corrected labels | Report the (small) difference honestly |
| H3 | Decomposition | Removing/normalising domain-identifying features explains part, but not all, of the collapse | Report "0% explained" as a real, publishable finding |
| H4 | Drift type | Gradual (temporal) drift raises false positives less sharply than a sudden network switch | Note the CIC-2017 day/attack-mix confound explicitly |
| H5 | Detectability | Our detector flags a network switch faster / cheaper than scheduled retraining | Compare cost, not just speed |
| H6 | Recovery | Test-time adaptation (Tent/AdaBN) and meta-learned fast adaptation recover more FPR@95%DR than CORAL/DANN, at lower labelling cost | An honest "adaptation doesn't help much here" is still reportable |
| H7 | Poisoning | An unguarded self-updating adapter is measurably degradable by slow poisoning; our safeguards reduce that degradation | Report the vulnerability even if the fix under-performs |
| H8 (stretch) | Extractor sensitivity | Re-extracting the *same* PCAPs with a different flow tool changes performance non-trivially | Optional if time runs out |

---

## Part 6 — Datasets (final list)

| Dataset | Created by | Role | Feature schema | Notes |
| --- | --- | --- | --- | --- |
| **LycoS-IDS2017** | Rosay et al., via LycoStand on CIC-IDS2017 raw PCAPs | Source (label-corrected) | CICFlowMeter-family, ~77-80 features | Use this, not raw CIC-IDS2017 |
| **LycoS-Unicas-IDS2018** | Cantone et al. 2024, via LycoStand on CSE-CIC-IDS2018 raw PCAPs | Target, same lab family (label-corrected) | Same schema as LycoS17 | Public on GitHub; use this, not raw CSE-CIC-IDS2018 |
| **NF-UNSW-NB15-v2** | Sarhan, Layeghy & Portmann (NetFlow-v2 conversion of UNSW-NB15) | Target, genuinely different lab | NetFlow-v2, 43 features | Different feature *vocabulary*, not just a different network — the harder test |
| **NF-ToN-IoT-v2** (optional/stretch) | Same NetFlow-v2 family | IoT target | NetFlow-v2, 43 features | Stretch goal if time allows |

**Why this exact set:** LycoS17/LycoS18 give us a clean, already-corrected, same-schema pair (mirrors and extends Cantone et al. exactly). NF-UNSW/NF-ToN give us the cross-*schema* generalisation Cantone et al. explicitly did not attempt — this is where our contribution is least contested.

**Class imbalance reminder** (from Cantone et al.'s Table 2, useful context): benign traffic is 73–83% of every one of these dataset families. Never report plain accuracy.

---

## Part 7 — Models

| Model | Why it's here |
| --- | --- |
| LDA | Cheapest possible baseline — surprisingly, the *best generalising* model in Cantone et al.'s study. Always include it; don't assume complexity helps. |
| Decision Tree | Simple, interpretable baseline |
| Random Forest | Strong, fast tabular baseline; widely used in NIDS literature |
| XGBoost | Strongest classical baseline; usually the ceiling for tree-based methods |
| MLP | Basic neural baseline — needed as a gradient-based model for later adaptation (CORAL/DANN/Tent all assume differentiable features) |
| Autoencoder | Anomaly-detection-style baseline, trained mostly on benign traffic |
| TabNet | Attention-based deep tabular model — chosen specifically because, unlike trees, it can be updated at test time via gradient methods (Tent) and is a natural fit for meta-learning (Part 9) |

---

## Part 8 — Metrics, Drift Detection, and Adaptation

### 8.1 Metrics

```
Detection Rate (Recall) = TP / (TP + FN)
FPR                     = FP / (FP + TN)
FPR @ 95% Detection Rate = the false-alarm rate an analyst tolerates
                            to catch 95 of every 100 attacks
```

Threshold is set on **source validation data only**, frozen, then applied unchanged to the target test set (never re-tuned on target — that would leak target labels). PR-AUC reported alongside as a threshold-free complement. MCC is kept as a secondary metric for direct comparability with Cantone et al.'s numbers.

**Rules:** 3 random seeds, mean ± std reported; test data used exactly once.

### 8.2 Drift detection

| Method | What it does | Needs labels? |
| --- | --- | --- |
| KS test per feature (Bonferroni-corrected) | Flags which individual features shifted | No |
| MMD (Maximum Mean Discrepancy) | Multivariate distance test, catches shifts a single-feature test misses | No |
| ADWIN on the stream of model confidence | Adaptive window that cuts when older vs. newer confidence differs | No |

Output is not just yes/no — it's a ranked list of which features shifted, plus a cost-sensitive trigger that only recommends action when the *expected* rise in false alarms justifies it.

### 8.3 Adaptation methods

| Method | Idea | Needs target labels? | Status |
| --- | --- | --- | --- |
| Per-domain feature scaling | Normalise each domain separately | No | Cheapest baseline |
| Deep CORAL (2016) | Match covariance of source/target feature activations | No | Committed baseline (Phase D) |
| DANN (2016) | Gradient-reversal domain classifier | No | Committed baseline (Phase D) |
| AdaBN | Re-estimate batch-norm stats on target traffic | No | Extension |
| Tent | Test-time entropy minimisation, updates only normalisation params | No | Extension, our main "live" bet |
| Few-shot labelling (1%/5%/10% target labels) | Analyst labels a small, detector-flagged sample | A little | Committed (Phase D) |
| **Meta-learned fast adaptation** | See Part 9 | No (trained offline on existing pairs) | **New — replaces RL** |

---

## Part 9 — Fine-Tuning: Why We Dropped RL, and What We Use Instead

*This section documents a design decision the team already made and recorded — every member should be able to justify it in the viva.*

### 9.1 Why not reinforcement learning

RL earns its keep when you must make sequential decisions against a delayed reward with no other way to get a training signal. We're not in that situation: (1) our offline logs — the collapse matrix and every adaptation experiment we run in Phase D — already tell us the outcome of each "action" (which adapter, applied to which drift situation, produced how much FPR@95%DR improvement), so there's no need to explore blindly; and (2) "fine-tune the model given a measured gap" is fundamentally a **supervised, gradient-based problem**, not a trial-and-error one. Using RL here would be solving an easier problem the hard way — harder to train reliably in one semester, harder to defend in a viva, and more novel-hence-risky than the result needs to be.

**Comparison of alternatives considered:**

| Technique | What it does | Fit here |
| --- | --- | --- |
| RL (policy gradient / Q-learning) | Trial-and-error against a reward signal | Sample-inefficient; rejected |
| Contextual bandit | Learns best action per context, no sequencing | Workable but still needs an exploration strategy we don't need |
| **Meta-learning (Reptile)** | Trains the base model so a few gradient steps on new data adapt it well | **Adopted** — matches our exact problem |
| **Supervised reward modeling** | Train a regressor to predict how much each adapter would help, then pick the best | **Adopted** — simplest correct tool |
| Self-training / pseudo-labelling | Retrain on the model's own confident predictions | Already implicit in our adaptation menu |
| Bayesian optimisation | Sample-efficient hyperparameter search | Useful *around* an adapter, not a replacement |

### 9.2 The two-part solution we're building

**Part A — "Which adapter, and how strong?" → Supervised reward modeling.**
Train a small regressor (gradient-boosted trees are enough) where the **input** is the live drift statistics (per-feature KS values, MMD statistic, prediction entropy, the trigger's estimated FPR rise) and the **output** is the measured FPR@95%DR improvement each adapter achieved for that kind of drift, taken from our own Phase B/D experiment logs. At run-time, feed in the live drift statistics and pick whichever adapter the regressor predicts will help most. No exploration/exploitation trade-off to tune, no regret bounds to justify — just ordinary supervised learning on data we're generating anyway.

```
Training data (from our own logs):
  state  = [KS_1, KS_2, ..., MMD, entropy, estimated_FPR_rise]
  action = which adapter was applied (CORAL / DANN / AdaBN / Tent / few-shot-k)
  reward = measured ΔFPR@95%DR after applying that adapter

Regressor: state, action → predicted reward
Run-time: pick action = argmax_action  regressor(state, action)
```

**Part B — "How do the model's weights actually update?" → Meta-learning (Reptile).**
Instead of hand-designing the adaptation rule, we train the base model (MLP or TabNet, since this needs gradients) **across our multiple source→target pairs** so it becomes explicitly good at adapting in a handful of gradient steps to a *new*, unseen target. Reptile is the practical choice over MAML because it needs no second-order gradients — it's a small addition on top of the existing training loop, and it degrades gracefully (if it doesn't beat Tent/AdaBN, that's still a clean, honest, reportable result).

```
Reptile sketch:
for each meta-training step:
    sample a (source, target) pair from our dataset collection
    clone the current model weights θ
    take k gradient steps on the source task → θ'
    update θ ← θ + ε (θ' − θ)      # move θ toward θ'
# At deployment: given a NEW target with drift detected,
# a few gradient steps from θ should adapt quickly and well
```

**Why this framing is a better story than adding RL as a fourth mechanism:** it turns H6 (recovery) into a clean three-way comparison — Tent/AdaBN vs. CORAL/DANN vs. meta-learned fast adaptation — with every piece trainable from data we already planned to collect. No live interaction loop, no reward-shaping headaches, and one fewer novel-and-therefore-risky thing to defend.

**New reading to add (Tier 1, for whoever owns adaptation):** look up "Learning to Adapt"-style meta-learning-for-domain-adaptation work — **MetaReg** and **ARM (Adaptive Risk Minimization)** are the two most citable — for the right vocabulary and a principled justification for meta-learning over RL, which is exactly what a panel will probe.

---

## Part 10 — Methodology, Timeline, Risks

### 10.1 Phases

| Phase | Window | What happens | Output |
| --- | --- | --- | --- |
| A — Harmonise data | 26 Sep – 9 Oct | Map datasets to shared schema; use LycoS/LycoS-Unicas corrected labels; drop IPs/timestamps/flow IDs; remove duplicates; time-based splits | Clean benchmark |
| B — Baselines | 10 – 19 Oct | Train RF, XGBoost, MLP on one dataset, test on all others; original vs. corrected labels (lighter than originally planned — see Part 2.3 strategic note) | Collapse matrix |
| C — Drift detection | 10 Oct – 13 Nov | KS per feature; MMD; ADWIN on confidence stream | Drift alarm v1 by Mid-Sem |
| D — Adaptation | 24 Oct – 13 Nov | Feature scaling; CORAL; DANN; few-shot (1/5/10%); **+ reward model + Reptile meta-learning** | Adaptation results |
| E — Live prototype | 14 – 29 Nov | tcpreplay → flow extractor → FastAPI model → dashboard; latency/load tests; poisoning test | Working demo |

### 10.2 Evaluation matrix (train ↓ / test →)

|  | CIC-17c | CIC-18c | NF-UNSW | NF-ToN |
| --- | --- | --- | --- | --- |
| **CIC-17c** | within | cross | cross | cross |
| **CIC-18c** | cross | within | cross | cross |
| **NF-UNSW** | cross | cross | within | cross |
| **NF-ToN** | cross | cross | cross | within |

Hypothesis to confirm: every cross-dataset cell shows a higher FPR than the within-dataset cell in its row (H1 — already independently confirmed by Cantone et al. on the CIC/LycoS side).

### 10.3 Full timeline

| Task | Window |
| --- | --- |
| Proposal & literature survey | Sep |
| Phase A — harmonisation | 26 Sep – 9 Oct |
| Phase B — baselines & collapse matrix | 10–19 Oct |
| **Mid-Sem evaluation** | 20–23 Oct |
| Phase C — drift detector | 10 Oct – 13 Nov |
| Phase D — adaptation + fine-tuning | 24 Oct – 13 Nov |
| Phase E — live pipeline & poisoning test | 14–29 Nov |
| Paper draft | 14 Nov – mid Dec |
| **TA2 evaluation** | 30 Nov – 4 Dec |
| Report & final checks | 5–22 Dec |
| **End-Sem evaluation** | 23–30 Dec |

### 10.4 Risks

| Risk | Mitigation |
| --- | --- |
| Data preparation takes longer than planned | Start immediately; use ready NF-v2 CSVs and the already-corrected LycoS/LycoS-Unicas files (no manual correction needed) |
| Live extractor features differ from training features | Lock one extractor in week 1; train the demo model on its output |
| Laptop memory limits | Stratified subsampling; Kaggle/Colab for heavy runs |
| Adaptation gives little improvement | Report it honestly — a clean negative result is still publishable |
| Meta-learning / reward model don't beat simpler baselines | Fine — report as a comparison, not a required win; Tent/AdaBN/CORAL/DANN still stand on their own |

### 10.5 Tools

- **ML:** Python, pandas, scikit-learn, XGBoost, PyTorch
- **Drift:** `river` (ADWIN), `alibi-detect` (KS, MMD)
- **Traffic:** Wireshark, tcpreplay, NFStream
- **Serving:** FastAPI, Streamlit, Docker
- **Tracking:** Git/GitHub, MLflow
- **Compute:** 16 GB laptops; Kaggle/Colab GPU as needed

---

## Part 11 — Reading Plan

Depth key: **Skim** = abstract + intro + conclusion (15–25 min). **Read** = full paper, viva-ready (60–90 min). **Study** = read + be able to reproduce the core idea in code.

### Tier 0 — everyone reads before writing code

| Paper | Depth | Why |
| --- | --- | --- |
| **Cantone, Marrocco & Bria, IEEE Access 2024** (the uploaded PDF) | **Study** | Our direct Phase 1 anchor — see Part 2 in full. Know every number in Part 2.2. |
| Sommer & Paxson, "Outside the Closed World" (2010, IEEE S&P) | Read | Foundational argument for why NIDS evaluation is hard — cite when asked "why does this matter" |
| Sharafaldin et al., CIC-IDS2017 (2018, ICISSP) | Read | How the source dataset was built |
| Engelen et al. (2021, IEEE SPW) | Read | The audit behind LycoS-IDS2017's corrections |
| Liu et al. (2022, IEEE CNS) | Read | Error-prevalence study; confirms CSE-CIC-IDS2018 also needed correcting |
| Apruzzese, Pajola & Conti, XeNIDS (2022, IEEE TNSM) | Read | Closest prior cross-evaluation work besides Cantone et al. — know exactly what it does and doesn't do |
| Arp et al., "Dos and Don'ts" (2022, USENIX Security, Distinguished Paper) | Skim | Evaluation-pitfalls checklist; self-audit tool |

### Tier 1 — role-specific deep reads

**Data / drift owner:**

- Rosay et al., LycoS-IDS2017 methodology paper — **Read** (you'll use their tool's output directly)
- Sarhan, Layeghy & Portmann, NF-v2 standard features (2022) — **Read**
- Bifet & Gavaldà, ADWIN (2007, SIAM SDM) — **Read**, short and clean
- Jordaney et al., Transcend (2017, USENIX Security) — **Read**
- Yang et al., CADE (2021, USENIX Security) — **Study** — closest "explain the drift" system; know precisely how ours differs (we adapt, CADE only explains)
- Garg et al., ATC / predicting OOD performance (2022, ICLR) — **Skim** — basis of our label-free trigger

**Models / adaptation / fine-tuning owner:**

- Sun & Saenko, Deep CORAL (2016) — **Read**
- Ganin et al., DANN (2016, JMLR) — **Read**
- Li et al., AdaBN (2016) — **Read**, short and directly implementable
- Wang et al., Tent (2021, ICLR) — **Study** — your main test-time method
- Liang et al., SHOT (2020, ICML) — **Skim**
- Arik & Pfister, TabNet (2021, AAAI) — **Read**
- **MetaReg / ARM (Adaptive Risk Minimization)** — **Read** — the meta-learning-for-domain-adaptation justification (new, see Part 9)
- Andresini et al., INSOMNIA (2021, ACM AISec) — **Read** — closest prior semi-supervised drift-adaptation NIDS work
- Yang et al., ReCDA (2024, ACM SIGKDD) — **Skim** — most recent comparable method; check its exact setting

**Systems / live pipeline / security owner:**

- Mirsky et al., Kitsune (2018, NDSS) — **Read** — lightweight real-time pipeline reference
- Apruzzese, Fass & Pierazzi, adversarial perturbations meet concept drift (2024, ACM AISec) — **Read** — directly informs the poisoning-test design
- Layeghy & Portmann, explainable cross-domain evaluation (2023) — **Skim**

### Tier 2 — skim only, for related-work context

CICIoT2023, netFound, E-GraphSAGE, the LLM-for-NIDS survey, "How Dataset Diversity Affects Generalization" (2025), "Navigating the Latent Manifold" (2026 preprint, unreviewed — don't lean on it) — 10–15 min each, split across the team.

**Total load estimate:** ~7 Tier-0 reads (all, including the anchor paper — this is non-negotiable), ~5–7 Tier-1 reads per subsystem owner, ~6 Tier-2 skims shared. Front-load this before Mid-Sem; reading time should drop sharply after the schema freeze.

---

## Part 12 — Learning Resources (tools, not papers)

- **Domain adaptation:** the DANN paper's gradient-reversal diagram (first few pages) is unusually readable; the CORAL paper's conclusion is a good one-page summary of its limits. The Transfer Learning Library (`dalib`, `dalib.readthedocs.io`) tutorial has working PyTorch code for CORAL/DANN-style losses that transfers directly to tabular data.
- **Concept drift / streaming ML:** `river` (formerly `creme` + `scikit-multiflow`, on GitHub as `online-ml/river`) has working ADWIN implementations to validate your own against. The PWPAE repository (`Western-OC2-Lab` on GitHub) is a full worked concept-drift tutorial on network/IoT data.
- **Adversarial validation:** "Adversarial Validation, Explained" (Zygmunt Zając, FastML/KDnuggets, two parts) and "Adversarial Validation Overview" (Zak Jost, KDnuggets, code-complete with CatBoost) are the standard practical write-ups.
- **mRMR feature selection:** Cantone et al.'s own citations (Ding & Peng 2005; Peng, Long & Ding 2005) are the original papers — read these if you're implementing mRMR rather than using a library.
- **TabNet:** pair the paper with a working implementation (`dreamquark-ai/tabnet` is the standard open-source one) rather than implementing from the paper alone.
- **Meta-learning:** the Reptile blog post/paper from OpenAI is the most implementation-friendly introduction — much shorter and more concrete than MAML's original paper.
- **MLOps:** MLflow's own "Quickstart" docs cover everything needed here (log params, metrics, a run ID per experiment).
- **Packet replay / flow extraction:** `tcpreplay` and `NFStream`/`CICFlowMeter` official READMEs cover what's needed; no deeper resource required unless you hit a specific bug.

---

## Part 13 — Team Roles

| Member | Responsibility | Deliverables | Papers owned |
| --- | --- | --- | --- |
| Data & drift | Dataset harmonisation, leakage checks, drift detector | Scripts, dataset card, Git commits | Engelen, Liu, Sharafaldin, Sarhan, Rosay, Bifet & Gavaldà |
| Models & adaptation | Baselines, collapse matrix, CORAL/DANN, few-shot, reward model, Reptile | Training code, MLflow runs | Catillo, Arp, Apruzzese, DANN, CORAL, AdaBN, Tent, MetaReg/ARM |
| Systems & demo | PCAP replay, flow extraction, API, dashboard, latency tests, poisoning test | Pipeline code, demo video | Sommer & Paxson, CADE, INSOMNIA, Kitsune |

**Shared by everyone:** the Cantone et al. 2024 paper (Part 2), the literature survey, the research paper draft, and the final report. Every member must be able to explain the full pipeline end to end, not just their own slice.

---

## Part 14 — Glossary

| Term | Meaning |
| --- | --- |
| Flow | A summary record of one "conversation" between two network endpoints |
| CICFlowMeter | The extraction tool behind the CIC-IDS datasets, ~80 features/flow |
| LycoStand | The corrected re-extraction tool used to build LycoS-IDS2017 and LycoS-Unicas-IDS2018 |
| NetFlow-v2 | A standard, shared 43-feature flow schema used across several newer datasets |
| Source / target domain | The dataset trained on / the new network the model is applied to |
| Collapse matrix | The train-row × test-column results table — the empirical core of the project |
| MCC | Matthews Correlation Coefficient — a balanced metric even under class imbalance |
| FPR@95%DR | False-positive rate when the threshold catches 95% of attacks |
| mRMR | Minimum Redundancy Maximum Relevance — a feature-selection method |
| ADWIN | Adaptive Windowing — a streaming change-point detector |
| AdaBN | Re-estimating batch-normalisation statistics on new (target) data |
| Tent | Test-time adaptation via entropy minimisation on the unlabelled stream |
| Reptile | A first-order meta-learning algorithm; trains a model to adapt fast with a few gradient steps |
| Supervised reward modeling | Training a regressor to predict an action's benefit from logged outcomes, then greedily picking the best action — our RL replacement |
| Config hash | A fingerprint of an experiment's settings, logged for traceability |

---

*This document supersedes the earlier "DriftGuard Technical Guide" and both individual v1.2/v1.3 drafts for anything they disagree on. It should be treated as the team's single source of truth going into Mid-Sem.*