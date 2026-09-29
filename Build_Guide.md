# Build Guide: Cross-Dataset NIDS under Drift

**What this is:** the build plan. It lists every component, the order to build them in, what each one must contain, and how to tell when it is done.
**Source of truth:** the Complete Project Guide (25 Sep 2026). If this file and that guide disagree on scope, the guide wins. If they disagree on how something is built, this file wins.
**Version:** 1.0, 28 Sep 2026

---

## 0. How to use this file

- There are 19 components, **C0 to C18**. Each one gets a GitHub issue labelled with its ID, and each branch is named after it (for example `c05-harness`).
- A component is done only when every box under **Done when** is ticked. The code must also be merged through a PR that someone else reviewed.
- Build in the order given in Section 1. Where two components have no dependency between them, they can run in parallel. For example, C13 (demo data) starts in week 1 alongside the data work.

---

## 1. Build order at a glance

```
C0 Repo ─► C1 Ingest ─► C2 Schema ─► C3 Clean/split ─► C5 Harness ─► C6 Results table ─► C7 Decomposition
                                          │                 │                                  ▲
                                          └─► C4 Adv. validation ──────────────────────────────┘
                                                            │
                                                            └─► C8 Drift monitor ─► C9 Adapters ─► C10 Adaptation logs ─► C11 Selector
                                                                                        └─► C12 Reptile
C13 Demo data ─► C14 Detector API ─► C15 Replay runner ─► C16 Dashboard ─► C17 Poisoning test
                                                     (C8, C9, C11, C12 plug into C14/C16)
All ─► C18 Reports & paper
```

| ID | Component | Owner | Window | Depends on | Main output |
|---|---|---|---|---|---|
| C0 | Repo & environment | Everyone (Systems leads) | 26–29 Sep | — | Repo, env, MLflow, conventions |
| C1 | Data ingest | Data & drift | 26 Sep – 2 Oct | C0 | Raw data + manifest + Parquet |
| C2 | Schema, feature tracks & labels | Data & drift | 29 Sep – 5 Oct | C1 | `processed/{dataset}/{track}.parquet` |
| C3 | Clean & split | Data & drift | 2–9 Oct | C2 | Clean data + frozen split files |
| C4 | Adversarial validation | Data & drift | 10–19 Oct | C3 | Domain AUC + ranked "lab-telling" features |
| C5 | Train/eval harness | Models & adaptation | 6–14 Oct | C3 | One CLI that trains, thresholds and scores |
| C6 | Results table (N×N matrix) | Models & adaptation | 12–19 Oct | C5 | Collapse matrix, H1/H2 results |
| C7 | Decomposition | Models & adaptation | 24 Oct – 6 Nov | C4, C6 | Share-of-collapse explained (H3), time drift (H4) |
| C8 | Drift monitor | Data & drift | 10 Oct – 13 Nov | C5 | `DriftReport` per window, H5 |
| C9 | Adapters | Models & adaptation | 24 Oct – 10 Nov | C8 | 6 adapters behind one interface |
| C10 | Adaptation logs | Models & adaptation | 1–10 Nov | C9 | Table of every adapter × every drift window |
| C11 | Adapter selector | Models & adaptation | 6–13 Nov | C10 | Regressor that picks the adapter or says wait |
| C12 | Reptile meta-learning | Models & adaptation | 3–13 Nov | C9 | Meta-trained MLP, H6 comparison |
| C13 | Demo data | Systems & demo | 26 Sep – 19 Oct | C0 | NFStream flows from CIC-IDS2017 PCAPs + labels + demo model |
| C14 | Detector API | Systems & demo | 20 Oct – 20 Nov | C13 | FastAPI service with active + candidate models |
| C15 | Replay runner | Systems & demo | 10–22 Nov | C14 | tcpreplay → NFStream → API, latency numbers |
| C16 | Dashboard | Systems & demo | 14–29 Nov | C15, C8, C11 | Side-by-side live view with approval gate |
| C17 | Poisoning test | Systems & demo | 20–29 Nov | C16, C9 | Guarded vs unguarded degradation (H7) |
| C18 | Reports & paper | Everyone | continuous; paper 14 Nov – mid Dec | all | Dataset card, report, paper, demo video |

**Critical path:** C1 → C2 → C3 → C5 → C6. This path must be finished by the **Mid-Sem evaluation on 20 Oct**, so nothing else is allowed to delay it.

---

## 2. Ground rules for every component

These are the rules that stop the results from being wrong or unrepeatable. They are not optional.

1. **Configs, not code edits.** Every run starts from a YAML file in `configs/`. Nothing gets hard-coded in scripts.
2. **Config hash.** Each run computes `sha1(json.dumps(cfg, sort_keys=True))[:10]` and logs it to MLflow together with the git commit. Every number in the report must trace back to one run ID.
3. **Seeds.** Every result uses three seeds, `[0, 1, 2]`, and is reported as mean ± std.
4. **Thresholds come from source validation only.** The threshold is never tuned on target data. The one exception is few-shot adaptation: labels bought within the label budget may be used, and that must be logged.
5. **Each test split is used once**, at the final evaluation for that config. Tuning uses validation splits only.
6. **Never report plain accuracy.** The headline metric is FPR@95%DR, reported alongside the DR actually achieved on the target. Also report PR-AUC, and MCC so the numbers compare with Cantone et al.
7. **No results from notebooks.** Notebooks are for exploration. Every table and figure comes from a script in `scripts/`.
8. **Data is never committed.** `data/` and `mlruns/` go in `.gitignore`. Files are identified by the checksums in `data/MANIFEST.csv`.
9. **Identifiers are never features.** That means flow ID, IP addresses, timestamps and source port. Destination port is kept, but it is flagged and tested in C4 and C7.

---

## 3. Repository layout (create in C0)

```
nids-drift/
├── configs/            data.yaml, tracks/, train/, adapt/, drift.yaml, nfstream.yaml, live.yaml
├── data/               raw/  interim/  processed/  splits/  MANIFEST.csv      (gitignored)
├── src/xnids/
│   ├── utils/          seed.py, config.py (load + hash), log.py (MLflow wrapper)
│   ├── data/           ingest.py, schema.py, labels.py, clean.py, split.py
│   ├── features/       tracks.py, core_map.py
│   ├── models/         zoo.py, mlp.py, ae.py, tabnet_wrap.py
│   ├── eval/           metrics.py, harness.py, matrix.py
│   ├── analysis/       advval.py, ablation.py, timedrift.py
│   ├── drift/          ks.py, mmd.py, adwin.py, explain.py, trigger.py, monitor.py
│   ├── adapt/          base.py, scaling.py, coral.py, dann.py, adabn.py, tent.py, fewshot.py, reptile.py
│   ├── select/         logs.py, selector.py
│   ├── live/           extract.py, registry.py, api.py, replay.py, store.py
│   └── attack/         poison.py, guards.py
├── dashboard/          app.py
├── scripts/            one CLI per step (ingest, build_tracks, split, train, matrix, advval, ...)
├── tests/
├── reports/            figures/  tables/
├── notebooks/          exploration only
├── Dockerfile, docker-compose.yml, Makefile, requirements.txt, README.md, DATASET_CARD.md
```

---

## 4. Shared contracts

Agree on these interfaces before building. Everything else plugs into them.

| Contract | Where | Contents |
|---|---|---|
| Processed flows | `data/processed/{dataset}/{track}.parquet` | feature columns (float32) + `y` (0/1) + `family` (8 families) + `row_id` + `ts` (if the dataset has one) |
| Split file | `data/splits/{dataset}.parquet` | `row_id`, `split` ∈ {train, val, test}, plus the split config hash |
| Model bundle | `models/{version}/` | `model.*`, `scaler.pkl`, `threshold.json`, `features.json`, `meta.json` (config hash, run ID, parent version, training data) |
| Scores | MLflow artifact `scores_{target}.parquet` | `row_id`, `score`, `y`, used by C6, C7, C10 without retraining |
| DriftReport | `drift/monitor.py` dataclass → JSON | `window_id`, `n`, `ks{feature: (stat, p)}`, `mmd(stat, p)`, `adwin_change`, `mean_conf`, `entropy`, `est_fpr_rise`, `top_features[5]`, `message`, `recommend` |
| Adaptation log row | `data/logs/adapt_log.parquet` | see C10 |
| API | `live/api.py` | see C14 |

---

# Phase A — Data (26 Sep – 9 Oct)

## C0. Repo & environment

**Goal:** everyone can run the same code with the same library versions and get the same numbers.

**Build**
1. Create the GitHub repo with the layout above. Protect `main` so every change needs a PR with one review.
2. Pin Python 3.11 and set up `requirements.txt` with pinned versions. It must include pandas, polars, pyarrow, scikit-learn, xgboost, lightgbm, torch, pytorch-tabnet, alibi-detect, river, scipy, nfstream, fastapi, uvicorn, streamlit, mlflow, pyyaml, pytest, ruff.
3. Write `utils/seed.py` (seeds python, numpy, torch and cudnn deterministic), `utils/config.py` (loads YAML, computes the hash), and `utils/log.py` (starts an MLflow run and logs the config, hash, git commit and seed).
4. Run the MLflow tracking server with a SQLite backend (`mlflow server --backend-store-uri sqlite:///mlflow.db`). Heavy runs go to Kaggle or Colab and log back to the same store, or export their run folder into it.
5. Add Makefile targets `setup`, `test`, `lint`, `ingest`, `tracks`, `split`, `matrix`, `api` and `dashboard`.
6. Add a Dockerfile with one image for the API and dashboard.

**Example config** (`configs/train/rf_cic77.yaml`)
```yaml
run:   {name: rf_cic77, seeds: [0, 1, 2]}
data:  {track: cic77, source: lycos17, targets: [lycos18], max_rows_per_split: 2000000}
model: {name: rf, params: {n_estimators: 200, n_jobs: -1, class_weight: balanced_subsample}}
eval:  {target_dr: 0.95, metrics: [fpr_at_dr, dr_at_thr, pr_auc, mcc]}
```

**Done when**
- [ ] `make setup && make test` passes on all three laptops and on Colab
- [ ] One dummy run is visible in MLflow with its config hash and git commit
- [ ] The README says how to set up in 10 minutes or less

---

## C1. Data ingest

**Goal:** get every dataset onto disk once, verified, in fast Parquet.

| Dataset | Source | Notes |
|---|---|---|
| LycoS-IDS2017 | https://lycos-ids.univ-lemans.fr/ | Label-corrected CIC-IDS2017 (LycoSTand) |
| LycoS-Unicas-IDS2018 | https://github.com/MarcoCantone/LycoS-Unicas-IDS2018 | Single `LycoS18_dataset.csv` (Google Drive), **13,691,268 rows**, 77 features + label, 14 classes |
| NF-UNSW-NB15-v2 | https://staff.itee.uq.edu.au/marius/NIDS_datasets/ | NetFlow-v2 (43 fields) |
| NF-CSE-CIC-IDS2018-v2 | same UQ page | **Bridge dataset**: same traffic as LycoS18, different extractor (see C2) |
| NF-ToN-IoT-v2 | same UQ page | Stretch target |
| Original CIC-IDS2017 CSVs | https://www.unb.ca/cic/datasets/ids-2017.html | Needed only for H2 (original vs corrected) |
| CIC-IDS2017 PCAPs | same CIC page | Only the days C13 needs, starting with Monday (benign) plus 1–2 attack days |

**Build**
1. Write `scripts/ingest.py --dataset <name>`. It downloads the data (`gdown` for Google Drive), computes sha256 and appends to `data/MANIFEST.csv` (name, file, sha256, bytes, rows, date).
2. Convert CSV to Parquet in streaming mode (`polars.scan_csv(...).sink_parquet(...)`) so 13.7M rows never has to fit in RAM. Downcast floats to float32.
3. Clean up column names: strip whitespace, lowercase, and replace spaces with `_`. Keep a `rename_map.json` for each dataset.
4. Save a profile for each dataset in `reports/tables/profile_{dataset}.csv` with row count, label counts, NaN/inf counts per column, and dtypes.

**Done when**
- [ ] The LycoS18 row count matches 13,691,268 exactly, and the other datasets match their published counts. Any mismatch is explained in the dataset card.
- [ ] A second machine re-running ingest gets identical checksums
- [ ] Label counts are saved (benign should be roughly 73–83%, as in Cantone et al.)

**Pitfalls:** CIC column names have leading spaces and duplicates (for example `Fwd Header Length` appears twice), so de-duplicate them explicitly. Check whether LycoS uses different label strings from the original CIC release.

---

## C2. Schema, feature tracks & labels

**Goal:** turn five differently-shaped datasets into comparable tables. **This is the most important design decision in the project.** Datasets with different feature vocabularies cannot be compared naively.

### The three feature tracks

| Track | Datasets | Features | What it answers |
|---|---|---|---|
| **A: `cic77`** | LycoS17 ↔ LycoS18 | Native LycoSTand features (~77), identifiers dropped | Same extractor, different network. Replicates and extends Cantone et al. |
| **B: `nf43`** | NF-UNSW-v2 ↔ NF-ToN-v2 ↔ NF-CSE-CIC-2018-v2 | Native NetFlow-v2 fields minus addresses and source port (~39 left) | Same extractor, genuinely different labs. Cantone et al. excluded NetFlow |
| **C: `core`** | All of the above | ~12–15 semantically equivalent features mapped across both families | Cross-family comparison (for example LycoS17 → NF-UNSW) |

**Building track C (`configs/tracks/core_map.yaml`).** Each row maps one core feature to its column in each family. Each row also records the unit and a caveat. Candidates:

| Core feature | CIC/LycoS column | NetFlow-v2 column | Caveat |
|---|---|---|---|
| duration_s | Flow Duration (µs) | FLOW_DURATION_MILLISECONDS | Convert units |
| pkts_fwd / pkts_bwd | Tot Fwd / Bwd Pkts | IN_PKTS / OUT_PKTS | Direction convention |
| bytes_fwd / bytes_bwd | TotLen Fwd / Bwd Pkts | IN_BYTES / OUT_BYTES | CIC counts payload, NetFlow counts IP bytes |
| bytes_per_pkt_fwd/bwd | derived | derived | More robust than raw bytes |
| protocol | Protocol | PROTOCOL | — |
| pkt_len_min / max | Pkt Len Min / Max | SHORTEST / LONGEST_FLOW_PKT | Check header inclusion |
| syn / fin / rst / psh / ack flags | flag counts > 0 | bits of TCP_FLAGS | Reduce both to presence bits |
| init_win_fwd / bwd | Init Fwd / Bwd Win Byts | TCP_WIN_MAX_IN / OUT | Initial vs max, so mark as approximate |
| dst_port | Dst Port | L4_DST_PORT | Flagged shortcut feature |

**Validate the mapping with the bridge.** LycoS18 and NF-CSE-CIC-IDS2018-v2 come from the **same PCAPs**. For each core feature, compare the two distributions (quantiles and KS). Any feature whose distributions disagree badly on identical traffic is an extractor artefact, so drop it or fix the unit. Record this check in the dataset card; it is part of H8.

### Label taxonomy
Keep a binary `y` (benign = 0) as the task label. Also add `family` for analysis and for building attack-mix tasks:

| Family | Maps from (examples) |
|---|---|
| Benign | BENIGN, Benign, normal |
| DoS | DoS Hulk/GoldenEye/Slowloris/Slowhttptest, dos |
| DDoS | DDoS, LOIC, HOIC, ddos |
| Recon | PortScan, Reconnaissance, scanning, Analysis |
| BruteForce | FTP-/SSH-Patator, Brute Force, password |
| WebAttack | Web Attack (XSS, SQLi, Brute), injection, xss |
| Bot/Backdoor | Bot, Backdoor, backdoor |
| Other | Infiltration, Exploits, Fuzzers, Generic, Shellcode, Worms, ransomware, mitm, Heartbleed |

Store the map in `configs/labels.yaml`. Any label that is **not in the map raises an error** rather than silently becoming "Other".

**Build**
- `data/schema.py`: expected columns and dtypes for each track, with a `validate(df, track)` function
- `features/tracks.py`: `build_track(dataset, track) -> DataFrame`, which writes `processed/{dataset}/{track}.parquet`
- `data/labels.py`: `to_binary`, `to_family`

**Done when**
- [ ] Every (dataset, track) file exists and passes `validate`
- [ ] The bridge-check table is saved, and the list of core features that were dropped is justified
- [ ] There are zero unmapped labels, and the family counts table is saved

---

## C3. Clean & split

**Goal:** leakage-free, frozen train/val/test splits.

**Build**
1. **Non-finite values:** replace `inf` with NaN, then impute the median. The median is fit on source train only and stored in the model bundle. Add a boolean `_was_inf` column only if it is used.
2. **Negative sentinels:** CIC uses `-1` for "not applicable", for example on window size. Keep it as a value, but document it.
3. **Duplicates:** drop exact duplicates of (features + label). For **conflicting duplicates** (same features, different label), drop all copies and count them in the dataset card.
4. **Constant columns:** drop columns that are constant in source train. Decide this per training run, not globally.
5. **Splits:**
   - Datasets with timestamps (LycoS) get a **time-block split**. Cut the data into 5-minute blocks and assign whole blocks to train/val/test (60/20/20), stratified so every attack family shows up in every split where possible. This stops near-identical neighbouring flows leaking across splits.
   - NF-v2 has no timestamps, so use a stratified random split and **state this limitation**. NetFlow v3 (with timestamps) is the optional upgrade.
   - When a dataset is a **target**, its train split acts as the *unlabelled pool* for adaptation, and the small labelled budget is drawn from it. Its test split is only for evaluation.
6. **Dev subsample:** a stratified subsample of at most 2M rows per split for laptop runs, recorded in the config. Final numbers use the full data where memory allows (Kaggle).
7. Save `data/splits/{dataset}.parquet` and never regenerate it after 9 Oct. If it must change, bump the version and re-run everything.

**Tests** (`tests/test_splits.py`)
- No `row_id` appears in two splits
- No identifier columns among the features
- Each split has at least one example of every family that has at least 30 rows in total

**Done when**
- [ ] The split files are committed by hash, and the dataset card lists rows removed at each step
- [ ] The tests pass

---

# Phase B — Measurement (10 – 19 Oct, due Mid-Sem)

## C5. Train/eval harness

(Built before C4 because C4 and everything after it reuse it.)

**Goal:** one command that trains any model on any source and scores it on every target with identical rules.

**Models** (`models/zoo.py`, common interface `fit(Xtr, ytr, Xval, yval)` and `score(X) -> P(attack)`)

| Model | Setup |
|---|---|
| LDA | StandardScaler + `LinearDiscriminantAnalysis` (Cantone et al.'s best generaliser, so always include it) |
| Decision Tree | `max_depth` chosen from {10, 20, None} on source val |
| Random Forest | 200 trees, `balanced_subsample` |
| XGBoost | `hist`, up to 1,000 rounds, early stopping on source val, `scale_pos_weight` |
| MLP (PyTorch) | log1p + StandardScaler → 256-128-64 with **BatchNorm1d** + ReLU + dropout 0.2 → 1 logit. Class-weighted BCE, Adam 1e-3, early stopping. BatchNorm is required for AdaBN and Tent. |
| Autoencoder | Trained on **benign source only**. Score = reconstruction error, rescaled to [0, 1] using source-val quantiles |
| TabNet | `pytorch-tabnet`, default n_d = n_a = 16, early stopping on source val |

Hyperparameter search uses **source val only**, on a small fixed grid with the same budget for every model.

**Metrics** (`eval/metrics.py`)
```python
def threshold_at_dr(y_val, s_val, dr=0.95):
    s_att = np.sort(s_val[y_val == 1])
    k = int(np.floor((1 - dr) * len(s_att)))
    return s_att[k]                      # flag if score >= t  -> catches >= 95% of val attacks

def rates(y, s, t):
    pred = s >= t
    tp = (pred & (y == 1)).sum(); fn = (~pred & (y == 1)).sum()
    fp = (pred & (y == 0)).sum(); tn = (~pred & (y == 0)).sum()
    return fp / (fp + tn), tp / (tp + fn)   # FPR, DR
```
For every (model, source, target, seed), report:
- **FPR at the frozen source threshold, plus the DR achieved on the target at that threshold.** This is the operational headline. Always report both numbers, because a low FPR means nothing if DR collapsed.
- **Oracle FPR@95%DR on the target.** This is a diagnostic only, and must be labelled as such. It separates "the ranking broke" from "the calibration shifted".
- PR-AUC, and MCC at the frozen threshold

**CLI**
```
python scripts/train.py --config configs/train/rf_cic77.yaml
```
The script loops over seeds and targets, logs everything to MLflow, and saves the model bundle and `scores_{target}.parquet`.

**Done when**
- [ ] `tests/test_metrics.py` passes on hand-computed toy cases
- [ ] All 7 models run end-to-end on the dev subsample of LycoS17 → LycoS18
- [ ] Within-dataset MCC on LycoS17 is in the ~90s. If it isn't, something is broken: compare with Cantone et al.'s 94.63% average.

---

## C6. Results table (the N×N matrix)

**Goal:** the Mid-Sem centrepiece, showing how much each model loses when the network changes.

**Build**
1. `scripts/matrix.py --track cic77|nf43|core` pulls runs from MLflow by config hash and builds, per model, a train (rows) × test (columns) matrix of mean ± std.
2. Figures:
   - a heatmap for each model and track
   - a within vs cross bar chart
   - a comparison row against Cantone et al.'s MCC (average 94.63% within vs 29.35% cross, worst pair LycoS17 → LycoS18 at 10.83%)
3. **H2 (label inflation):** train and test on the original CIC-IDS2017 CSVs vs LycoS17.
   > **Confound:** the original release uses CICFlowMeter features and LycoS uses LycoSTand, so the difference is *extractor + labels*. Report it honestly as "original release vs corrected release" and do not claim it isolates label errors.
4. Export `reports/tables/matrix_{track}.csv` and a LaTeX version.

**Done when**
- [ ] The cic77 matrix is complete for all 7 models × 3 seeds
- [ ] The nf43 matrix is complete for at least LDA, RF, XGB and MLP
- [ ] The core-track matrix has at least RF and MLP
- [ ] H1 is stated with numbers, and H2 is stated together with the confound note

---

## C4. Adversarial validation

**Goal:** find which features simply "tell the lab apart". These are the ones preprocessing might fix.

**Build** (`analysis/advval.py`)
```python
def domain_auc(Xs, Xt, seed=0, n=200_000):
    Xs = Xs.sample(min(n, len(Xs)), random_state=seed); Xt = Xt.sample(min(n, len(Xt)), random_state=seed)
    X = pd.concat([Xs, Xt]); d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63)
    p = cross_val_predict(clf, X, d, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    imp = pd.Series(clf.fit(X, d).feature_importances_, X.columns).sort_values(ascending=False)
    return roc_auc_score(d, p), imp
```
1. Run it on **benign traffic only** for every source/target pair. This way the classifier learns lab differences rather than attack-mix differences. Then run it on all traffic as well.
2. **Iterative removal:** drop the top feature, recompute the AUC, and repeat until AUC < 0.7 or 10 features have been removed. Save the AUC-vs-removed curve.
3. Output `reports/tables/advval_{pair}.csv`, containing the ordered removal list and the AUC after each step.

**Done when**
- [ ] AUC curves exist for every pair in each track
- [ ] The top-k lab-telling features are listed for each pair and passed to C7

**Pitfall:** between different labs the AUC may stay close to 1.0 even after 10 features are removed. That is a finding, not a bug; report it.

---

## C7. Decomposition (H3 and H4)

**Goal:** quantify *why* performance collapses.

**Build**
1. **Ablation:** retrain each task model (RF, XGB, MLP at minimum) without the top-k lab-telling features from C4, for k ∈ {1, 3, 5, 10}.
2. **Share of collapse explained:**
   `share = (FPR_cross − FPR_cross_ablated) / (FPR_cross − FPR_within)`
   Report the raw value even if it is negative or greater than 1, and explain it.
3. **Normalisation variant:** per-domain rank or quantile transform instead of removal. This overlaps with the per-domain scaling adapter in C9, so reuse that code.
4. **Time drift (H4):** within LycoS17, train on early windows and test on later ones (gradual drift). Compare with a sudden switch to LycoS18.
   > Note the confound: CIC-2017 attack types change by day, so "time" and "attack mix" are entangled. Also report benign-only FPR over time.

**Done when**
- [ ] The share-explained table exists for each (pair, model, k)
- [ ] A plot of FPR vs time exists for the gradual and sudden cases
- [ ] H3 and H4 are written up with their honest outcome

---

# Phase C — Drift detection (10 Oct – 13 Nov)

## C8. Drift monitor

**Goal:** a label-free detector that says *whether* drift happened, *which* features shifted, and *whether it's worth acting on*.

**Build** (`drift/`)

| Piece | Implementation | Default settings |
|---|---|---|
| Reference | 20,000 flows sampled from source val | fixed seed |
| Windows | Consecutive windows of incoming flows | 5,000 flows |
| KS per feature | `scipy.stats.ks_2samp` | α = 0.05 / n_features (Bonferroni) |
| MMD | `alibi_detect.cd.MMDDrift`, RBF kernel, on standardised features (PCA to 20 dims if slow) | 1,000–2,000 samples, permutation p-value |
| ADWIN | `river.drift.ADWIN` on the stream of model confidence `max(p, 1−p)` | δ = 0.002 |
| Explain | Rank features by KS statistic and write a plain sentence for the top 5, for example "flow duration is 3.1× longer than in training" | template in `explain.py` |
| Cost trigger | Estimate the target error without labels using **ATC (Garg et al.)**, then turn it into expected extra false alarms per hour | recommend acting only if (extra alerts/hour × analyst cost) > adaptation cost |

Output is one `DriftReport` per window (Section 4). This same object feeds the dashboard (C16) and the selector (C11).

**Evaluation (H5)**
1. **No-drift stream** (shuffled source test): measure the detector's false-alarm rate per 100 windows.
2. **Sudden switch** at a known flow index: measure the detection delay in flows.
3. **Gradual ramp** (target share rising from 0 to 100%): measure the delay.
4. Compare against **scheduled retraining** every N flows on cost (labels + compute), not only speed.

**Done when**
- [ ] `monitor.run(stream) -> list[DriftReport]` works offline on any track
- [ ] A table of false-alarm rate and detection delay exists for KS, MMD, ADWIN and the combined trigger
- [ ] Explanations are sanity-checked: the top-ranked features overlap with C4's lab-telling list
- [ ] **Mid-Sem:** v1 with KS only, shown on the LycoS17 → LycoS18 switch

---

# Phase D — Adaptation (24 Oct – 13 Nov)

## C9. Adapters

**Goal:** six ways to repair a drifted model, all behind one interface so C10, C11 and C16 can call any of them.

```python
class Adapter:
    name: str
    needs_labels: bool
    def adapt(self, model, src_train, tgt_unlabelled, tgt_labelled=None) -> "ModelBundle": ...
```

| Adapter | Build notes |
|---|---|
| **Per-domain scaling** | Refit the scaler (and median imputer) on the target unlabelled pool. Works for every model. |
| **Deep CORAL** | Train the MLP with `BCE + λ·CORAL(h_s, h_t)` on the penultimate layer. `λ ∈ {0.1, 1, 10}`, fixed in advance; report all three. |
| **DANN** | Gradient reversal layer plus domain head on the penultimate layer, with λ schedule `2/(1+exp(−10p)) − 1` from the paper |
| **AdaBN** | Reset the BN running stats and recompute them on target batches. No gradients. |
| **Tent** | Train only the BN affine parameters to minimise prediction entropy on target batches. **Guard:** stop if the predicted attack rate moves more than 2× from source, because entropy minimisation can collapse to "everything benign". |
| **Few-shot** | Budgets of 50 / 200 / 1,000 labelled target flows (and 1 / 5 / 10%). Selection: random vs uncertainty vs drift-guided. MLP: fine-tune the last two layers. Trees: retrain on source + target with the target upweighted. The threshold may be re-picked on the labelled budget, and this must be logged. |

```python
def coral(hs, ht):
    d = hs.size(1)
    return ((torch.cov(hs.T) - torch.cov(ht.T)) ** 2).sum() / (4 * d * d)

@torch.no_grad()
def adabn(model, loader):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats(); m.momentum = None; m.train()   # only BN in train mode
    for x in loader: model(x)
    model.eval(); return model

def tent_step(model, opt, x):   # opt holds only BN weight/bias; dropout in eval mode
    p = torch.sigmoid(model(x).squeeze(-1)).clamp(1e-6, 1 - 1e-6)
    ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
    opt.zero_grad(); ent.backward(); opt.step()
    return p.detach()
```

**Rule:** the hyperparameters of unsupervised adapters are fixed before seeing target results, or chosen by source-only criteria. Never pick λ by target test FPR.

**Done when**
- [ ] Every adapter runs on cic77 LycoS17 → LycoS18 and on at least one nf43 pair
- [ ] A before/after table exists (FPR at frozen threshold, DR, PR-AUC) per adapter
- [ ] Unit tests: AdaBN changes the BN stats; Tent only changes BN parameters; CORAL loss is 0 for identical inputs

---

## C10. Adaptation logs

**Goal:** the training data for the selector. Every adapter is tried on every drift situation, so the logs are **full-information** (the outcome of every action is known). This is why a supervised regressor is enough and no RL or bandit exploration is needed.

**Build**
1. **Scenarios:** for each ordered pair in each track, sample about 50 target *windows* by varying time block, attack-mix resampling (use `family`) and window size.
2. For each window, run C8 **before** adapting to get the drift features.
3. Apply every action:
   - wait (Δ = 0)
   - scaling
   - AdaBN
   - Tent
   - CORAL
   - DANN
   - few-shot-{50, 200, 1000}
   - Reptile-{50, 200, 1000} (once C12 exists)

   CORAL and DANN are expensive, so train them **once per pair** and evaluate per window.
4. Measure the outcome on a held-out part of the same window.

**Row schema**

| Column | Meaning |
|---|---|
| `scenario_id`, `pair`, `track`, `window_id`, `seed` | Identity. `pair` is used for grouping in C11 |
| `ks_max`, `ks_mean`, `n_ks_sig`, `mmd_stat`, `mmd_p`, `adwin_flag`, `mean_conf`, `entropy`, `est_fpr_rise`, `attack_share_pred` | Drift features, known at decision time |
| `adapter`, `params` | The action |
| `labels_used`, `compute_s` | Cost |
| `fpr_before`, `fpr_after`, `dr_before`, `dr_after` | Outcome |
| `delta_fpr` = `fpr_before − fpr_after` | Positive = improvement |

**Done when**
- [ ] At least 2,000 rows, covering every track and adapter
- [ ] A sanity plot of mean `delta_fpr` per adapter per track exists

---

## C11. Adapter selector

**Goal:** given a live `DriftReport`, predict which adapter will help most after label cost, or say "wait".

**Build** (`select/selector.py`)
```python
y = logs.delta_fpr - lam * logs.labels_used * cost_per_label        # net utility
X = logs[drift_cols + ["adapter"]]                                  # adapter as categorical
reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05)
# evaluate with GroupKFold(groups=logs.pair)  -> leave-one-pair-out

def choose(report, adapters, margin=0.005):
    rows = pd.DataFrame([{**report.features(), "adapter": a} for a in adapters])
    pred = reg.predict(rows); i = pred.argmax()
    return ("wait", 0.0) if pred[i] < margin else (adapters[i], pred[i])
```
- **Uncertainty for the dashboard:** a 5-model bootstrap ensemble. Show mean ± std of the predicted gain.
- **Baselines:** always-Tent, the best single fixed adapter, random, and the oracle (best per scenario).
- **Metrics on held-out pairs:**
  - mean regret (oracle utility − chosen utility)
  - top-1 agreement with the oracle
  - Spearman correlation of predicted vs actual gain

**Done when**
- [ ] The leave-one-pair-out regret table vs baselines exists
- [ ] `choose()` is callable from the API (C14)
- [ ] The result is stated honestly. If the selector does not beat the best fixed adapter, report that.

---

## C12. Reptile meta-learning

**Goal:** train the MLP so that a few gradient steps on a *small labelled support set from a new target* adapt it well.

**Important correction to the project-guide sketch.** The inner steps run on the **target task's support set**, not on the source task. Each task = (pair, target window, small labelled support set S, query set Q).

```python
def reptile(model, sample_task, iters=3000, k=8, inner_lr=1e-3, eps0=0.1, meta_batch=4):
    for it in range(iters):
        eps = eps0 * (1 - it / iters)
        base = [p.detach().clone() for p in model.parameters()]
        delta = [torch.zeros_like(p) for p in base]
        for _ in range(meta_batch):
            task = sample_task()                                   # from TRAINING pairs only
            fast = copy.deepcopy(model); opt = torch.optim.SGD(fast.parameters(), lr=inner_lr)
            for _ in range(k):
                xb, yb = task.support_batch()                      # labelled target flows
                loss = F.binary_cross_entropy_with_logits(fast(xb).squeeze(-1), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            for d, pf, pb in zip(delta, fast.parameters(), base):
                d += (pf.detach() - pb) / meta_batch
        with torch.no_grad():
            for p, d in zip(model.parameters(), delta): p += eps * d   # θ ← θ + ε(θ̃ − θ)
    return model
```

**Build notes**
- **Where it can run:** a shared feature space with at least 3 domains, meaning track B (`nf43`, 3 datasets) and track C (`core`). Track A has only 2 datasets, so leave-one-out is impossible there.
- **Not enough pairs, so make many tasks per pair:** vary the time window, attack-mix resampling and support size (50 / 200 / 1,000). Also add synthetic domains by applying random per-feature scaling (×0.5–2) to source data.
- **Support sets** use the same selection rule as few-shot (C9), so there are at least some attacks in each.
- **BN statistics are not meta-learned**, so run AdaBN on the target before the k steps at deploy time.
- **Evaluation:** leave one target out. Meta-train on all other pairs, then at test time take k steps on the held-out target's support set and score its test set. Compare against:
  - plain fine-tuning from the source-trained MLP with the **same budget and steps** (the fair baseline)
  - few-shot (C9)
  - Tent and AdaBN (which need no labels)
  - CORAL and DANN

**Done when**
- [ ] The H6 table exists: FPR/DR after adaptation vs label budget, for each method and held-out target
- [ ] Reptile is registered as an action in C10 and C11
- [ ] A test shows one meta-iteration moves the parameters toward the fast weights

---

# Phase E — Live system (13 Oct – 29 Nov; C13 starts in week 1)

## C13. Demo data

**Goal:** a demo model trained on **exactly the features the live extractor produces**. The biggest risk in the project is a mismatch between the offline and live feature pipelines.

**Build**
1. **Lock the extractor in week 1:** NFStream with fixed settings in `configs/nfstream.yaml` (`idle_timeout`, `active_timeout`, `statistical_analysis: true`). C13 and C15 must read the same file.
2. Download the CIC-IDS2017 PCAPs for Monday (benign) and one or two attack days. If disk is tight, cut them to slices with `editcap -A/-B`.
3. Run `NFStreamer(source=pcap, **cfg).to_pandas()` and save the result as Parquet: this is track `nfs`.
4. **Labelling:** copy the published attacker/victim IPs and attack time windows from the CIC-IDS2017 page into `configs/cic17_attacks.yaml`:
   ```yaml
   - {family: BruteForce, name: FTP-Patator, day: Tuesday, start: "09:20", end: "10:20",
      attacker: [<from CIC page>], victim: [<from CIC page>]}
   ```
   A flow is an attack if its endpoints match an attacker–victim pair and it falls inside the window. Everything else is benign.
   > **Timezone check:** the published times are local time, while PCAP timestamps are epoch/UTC. Plot flow counts from the attacker IP over time to find the offset before labelling.
5. Train the demo model (MLP, so AdaBN and Tent work) on the `nfs` track with the C5 harness. Hold out time slices for replay.
6. **Drift segment for the demo:** re-extract a second PCAP from a *different network* with the same NFStream config. Prefer a CSE-CIC-IDS2018 day, which is labelled with the same IP/time scheme; UNSW-NB15 PCAPs with their ground-truth CSV are the alternative. Label it the same way.
7. Build `replay/demo.pcap` = segment A (in-distribution, about 5 min) followed by segment B (new network, about 5 min), using `mergecap` or sequential replay. Store the labels keyed by the 5-tuple and start time.
8. **Stretch (H8):** compare NFStream vs LycoS features on the same CIC-2017 PCAPs and train/test across them.

**Done when**
- [ ] Label counts per family match the published attack schedule within reason
- [ ] The demo model's within-data FPR@95%DR is reasonable, and there is a clear FPR jump on segment B
- [ ] `nfstream.yaml` is frozen and hashed

---

## C14. Detector API

**Goal:** a service that scores flows with the **active** model and a **candidate** model side by side, and supports promote and rollback.

**Endpoints** (`live/api.py`, FastAPI)

| Method | Path | Does |
|---|---|---|
| POST | `/score` | Batch of flows → scores + alerts from `active` and `candidate` (if present) |
| GET | `/health` | Status, loaded versions |
| GET | `/models` | Active, candidate, history |
| POST | `/drift/report` | Receives a `DriftReport` from the monitor, stores it, and calls selector `choose()` |
| GET | `/drift/latest` | Latest report + recommendation |
| POST | `/adapt` | Runs the chosen adapter on the buffered recent flows → registers the **candidate** (never auto-promotes) |
| POST | `/models/promote` | Needs `approved_by` + reason. Candidate becomes active; logged |
| POST | `/models/rollback` | Reverts to the previous active version; logged |
| GET | `/metrics` | Rolling alerts/min, flows/s, p50/p99 latency |

```python
class Flow(BaseModel):
    flow_key: str
    features: dict[str, float]

@app.post("/score")
def score(batch: list[Flow]):
    X = to_frame(batch, FEATURES)          # 422 if any feature is missing or extra
    return {role: registry.get(role).predict(X) for role in ("active", "candidate") if registry.get(role)}
```
- **Registry:** `models/{version}/` bundles (Section 4). The candidate stays loaded until it is promoted or rejected.
- **Store:** SQLite `live.db` with tables `scores`, `drift`, `actions` (audit log). The dashboard reads it.
- **Promotion gate checks** (run automatically before the promote button is enabled):
  - canary-set DR ≥ 0.9 × active model's canary DR
  - parameter change within the limit
  - predicted attack rate within bounds

**Done when**
- [ ] `tests/test_api.py` passes (422 on a bad schema, promote needs approval, rollback restores the version)
- [ ] p50 and p99 latency are reported for batches of 1, 100 and 1,000 flows

---

## C15. Replay runner

**Goal:** replay real traffic, extract flows live, and stream them to the API at a controlled rate.

**Build**
1. **Live mode (demo realism):** create a veth pair with `ip link add veth0 type veth peer name veth1`, replay with `tcpreplay -i veth0 --multiplier=<x> demo.pcap`, and capture with `NFStreamer(source="veth1", **cfg)`. This needs root or `CAP_NET_RAW`; in Docker use `--net=host --cap-add=NET_RAW --cap-add=NET_ADMIN`.
2. **File mode (reproducible experiments and fallback):** run NFStream on the PCAP file and send flows to the API at a set rate. The results should match live mode, so check that.
3. Micro-batch flows every 200 ms to `/score`, and send each 5,000-flow window to the drift monitor.
4. **Load test:** push the replay rate up (for example 1k, 5k, 10k, 20k flows/s, using `--topspeed` for the ceiling). Report the **sustained flows/s at which p99 < 100 ms** on a laptop.

**Done when**
- [ ] The whole `demo.pcap` plays end to end in both modes
- [ ] The latency/throughput table and plot are saved

**Pitfall:** NFStream emits a flow only when it expires (idle or active timeout), so alerts lag the packets. Show the lag on the dashboard, and keep the timeouts identical to C13.

---

## C16. Dashboard

**Goal:** the memorable demo. The same traffic is scored by the original and adapted models, side by side, with a human in control.

**Pages** (Streamlit, polling `live.db` or `/metrics` every 2 s)
1. **Live:**
   - flows/s
   - alerts/min for **original vs adapted**, side by side
   - FPR and DR computed from the known replay labels, marked "evaluation overlay: labels known because this is a replay"
2. **Drift:**
   - top-shifted features (KS bar chart)
   - MMD p-value over time
   - ADWIN change markers
   - the plain-language explanation
3. **Adapt:**
   - selector recommendation with predicted gain ± uncertainty
   - the gate-check results
   - **Approve / Reject / Rollback** buttons, which call the API
4. **Audit log:** every drift alert, recommendation, approval and rollback, with who did it and when.

**5-minute demo script**
| Time | What happens |
|---|---|
| 0:00 | Segment A replays; both models are quiet and FPR is low |
| 1:30 | Switch to segment B; the original model's FPR climbs |
| 2:00 | Drift alert with plain-language reason ("flow duration 3× longer…") |
| 2:30 | Selector recommends, for example, AdaBN; the candidate is built; gate checks pass |
| 3:00 | Analyst approves; adapted FPR drops while the original stays high |
| 4:00 | Show rollback, then the audit log |

**Done when**
- [ ] The demo script runs start to finish three times in a row without intervention
- [ ] A backup screen recording of the demo is saved

---

## C17. Poisoning test (H7)

**Goal:** show that an unguarded self-updating adapter can be steered by an attacker, and that the safeguards reduce the damage.

**Threat model:** the attacker can inject a fraction ρ ∈ {1%, 5%, 10%} of the traffic that the unsupervised adapter (Tent, AdaBN) learns from. They cannot touch labels or the model directly. **All experiments are offline, on replayed data.**

**Attacks** (`attack/poison.py`)
1. **Slow drift ("frog-boiling"):** over many adaptation rounds, inject flows of a chosen attack family whose features move a little toward benign each round. Tent's confidence-sharpening can then reinforce "benign" on that family.
2. **Statistic skew:** inject flows with extreme values in the top-KS features to drag the BN mean and variance.

**Safeguards** (`attack/guards.py`)
- Human approval gate (C14)
- **Canary set:** fixed labelled source and known-attack flows. Reject any update where canary DR drops by more than 5 points
- **Update-size limit:** clip the L2 distance between the new and old parameters
- **Robust statistics:** trimmed mean and variance for AdaBN (drop the top and bottom 1%)
- **Rollback**

**Measure:** DR on the targeted family and overall FPR per round, for unguarded vs each guard vs all guards, with 3 seeds.

**Done when**
- [ ] A plot of targeted-family DR vs rounds exists (unguarded vs guarded) for both attacks and each ρ
- [ ] H7 is written up with its honest outcome

---

# Reporting

## C18. Reports & paper

**Build throughout, not at the end.**
- `DATASET_CARD.md` covering sources, checksums, tracks, the core map with the bridge check, label map, rows removed, split scheme, and known confounds (H2 extractor, H4 day/attack mix, no timestamps in NF-v2)
- A model card for each model: data, features, threshold, known failure cases
- `scripts/make_tables.py` and `scripts/make_figures.py`, which regenerate **every** table and figure from MLflow run IDs
- Report chapters, filled as components finish: Intro, Literature, Data, Method (tracks, metrics), Results (H1–H8), Live system, Security (poisoning), Limitations, Conclusion
- Paper in IEEE template. Core story: cross-schema measurement + decomposition + learned adapter choice + poisoning of the adaptation loop. Submit only after supervisor approval.
- Demo video (≤ 5 min) and a one-command reproduction guide (`make reproduce`)

---

## 5. Milestone checklists

### Mid-Sem (20–23 Oct, 50 marks)
- [ ] C0–C3 done; dataset card v1
- [ ] C5 harness done; C6 matrix for cic77 (all models) and nf43 (≥ 4 models)
- [ ] C4 first AUC curves for LycoS17 ↔ LycoS18
- [ ] C8 v1: KS drift on the offline LycoS17 → LycoS18 switch
- [ ] C13: NFStream locked, CIC-2017 PCAPs extracted and labelled
- [ ] Slides: matrix heatmap, within-vs-cross gap vs Cantone et al., plan for Phases C–E

### TA2 (30 Nov – 4 Dec, 25 marks)
- [ ] C7–C12 done with result tables for H3–H6
- [ ] C14–C16 live demo working; C17 results for H7
- [ ] Paper draft (intro, method, main results)
- [ ] Backup demo video

### End-Sem (23–30 Dec, 100 marks)
Marks: Implementation 35, Report 21, Publication 21, Design 10, Significance 5, Ownership 4, Viva 4.
- [ ] Full report; every number traceable to a run ID
- [ ] Paper submitted (with evidence) after supervisor approval
- [ ] `make reproduce` works on a clean machine
- [ ] Each member can explain the full pipeline, not just their own part

---

## 6. Things that will go wrong (and what to do)

| Trap | Fix |
|---|---|
| Live features ≠ training features | One `nfstream.yaml`, used by both C13 and C15. The API rejects any mismatched schema with a 422. |
| Threshold quietly tuned on target | Only `threshold_at_dr(source_val)` exists in the harness; the target is scored after the threshold is frozen |
| Identifiers leaking in | Schema test fails if IP, flow ID, timestamp or source port appear in the features |
| Split leakage through near-duplicate flows | Time-block splits plus dedup before splitting |
| Laptop runs out of memory on 13.7M rows | Polars streaming, float32, dev subsample; full runs on Kaggle |
| Tent collapses to "all benign" | Attack-rate guard in C9, plus canary check in C14 |
| Few pairs for Reptile / selector | Many windows per pair, synthetic scaling domains, leave-one-pair-out evaluation |
| Adaptation barely helps | Report it. A clean negative result with a solid measurement is still publishable |
| Demo breaks on the day | File-mode replay fallback + recorded video |
| H2 over-claimed | Always attach the extractor confound note |

---

## Key links
- Cantone, Marrocco & Bria (2024), *IEEE Access* 12: https://ieeexplore.ieee.org/document/10704637/ (arXiv: https://arxiv.org/abs/2402.10974)
- LycoS-IDS2017: https://lycos-ids.univ-lemans.fr/
- LycoS-Unicas-IDS2018: https://github.com/MarcoCantone/LycoS-Unicas-IDS2018
- NetFlow datasets (UQ): https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- CIC-IDS2017 (CSVs, PCAPs, attack schedule): https://www.unb.ca/cic/datasets/ids-2017.html
- Deep CORAL: https://arxiv.org/abs/1607.01719 · DANN: https://arxiv.org/abs/1505.07818 · AdaBN: https://arxiv.org/abs/1603.04779 · Tent: https://arxiv.org/abs/2006.10726
- Reptile: https://arxiv.org/abs/1803.02999 · ARM: https://arxiv.org/abs/2007.02931 · ATC (Garg et al.): https://arxiv.org/abs/2201.04234
- NFStream: https://www.nfstream.org/docs/ · tcpreplay: https://tcpreplay.appneta.com/ · alibi-detect: https://docs.seldon.io/projects/alibi-detect/en/stable/ · river: https://riverml.xyz/ · MLflow: https://mlflow.org/docs/latest/index.html · FastAPI: https://fastapi.tiangolo.com/
