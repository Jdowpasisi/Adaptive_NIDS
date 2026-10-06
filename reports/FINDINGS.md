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
