# Dataset card: DriftGuard benchmark

Status: v2 (C1 ingest, C2 tracks and labels, C3 cleaning and frozen splits).
Checksums of every source file are in `data/MANIFEST.csv`. Per-dataset profiles are in `reports/tables/`.

## Sources

| Name | Dataset | Source | Published rows | Notes |
|---|---|---|---|---|
| `lycos17` | LycoS-IDS2017 | [lycos-ids2017 repo archive](https://maupiti-git.univ-lemans.fr/lycos/lycos-ids2017) | not published as one number | Ships **unlabelled** LycoSTand CSVs (one per weekday). Labels come from our port of the authors' `labelling.py` (`src/xnids/data/labellers.py`). |
| `lycos18` | LycoS-Unicas-IDS2018 | [GitHub](https://github.com/MarcoCantone/LycoS-Unicas-IDS2018), Google Drive id `12dQPcqRDJFeJGmqqshzdYN8M5t5tMXbQ` | 13,691,268 | 77 features + label. **No flow id, addresses, source port or timestamp.** |
| `nf_unsw_v2` | NF-UNSW-NB15-v2 | [UQ NIDS datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 2,390,275 | NetFlow v2, 43 fields. Manual download. |
| `nf_cse_cic18_v2` | NF-CSE-CIC-IDS2018-v2 | UQ | 18,893,708 | Bridge dataset: same PCAPs as LycoS18, different extractor. Manual download. |
| `nf_ton_v2` | NF-ToN-IoT-v2 | UQ | 16,940,496 | Stretch target. Manual download. |
| `cic17_orig` | CIC-IDS2017 (original CICFlowMeter CSVs) | shipped inside the LycoS17 archive | 2,830,743 | H2 only. |

## LycoS-IDS2017 labelling

The authors' script labels flows by attacker/victim address and attack time window per weekday. It also drops:
- all of Thursday afternoon, which holds the infiltration traffic;
- Friday flows to `205.174.165.73` after the bot window (portscan traffic wrongly labelled upstream);
- Friday flows to `52.6.13.28` and `52.7.235.158`.

Unit tests in `tests/test_ingest.py` check the rules, including their overwrite order.

**Verified:** our port gives 1,837,498 rows. Every per-label count matches the authors' own `python_logs/labelling_info.log` shipped in the archive exactly. Examples: benign 1,395,675; portscan 160,106; dos_hulk 158,988; ddos 95,683; heartbleed 11.

## Feature tracks (C2)

Processed files are `data/processed/{dataset}/{track}.parquet`. Each holds `row_id`, `ts` (LycoS17 and
cic17_orig only, epoch µs), `y`, `family`, the raw `label`, then the track's float32 features.
A dataset's `row_id` is the same in every track, so one split file per dataset serves all of its tracks.

| Track | Datasets | Features | Notes |
|---|---|---|---|
| `cic77` | lycos17, lycos18 | 77 native LycoSTand | Identical column sets in both datasets. `dst_port` flagged. |
| `nf43` | nf_unsw_v2, nf_cse_cic18_v2, nf_ton_v2 | 39 native NetFlow-v2 | Drops both addresses, `l4_src_port`, and `dns_query_id` (a random per-transaction id). `l4_dst_port` flagged. |
| `core` | all five above | 15 mapped | See below. `dst_port` flagged. |
| `cic_orig` | cic17_orig | 78 native CICFlowMeter | H2 only. `fwd_header_length_2` dropped as an exact duplicate (verified: 0 differing rows). |

### Core map and the bridge check (H8 evidence)

`configs/tracks/core_map.yaml` maps each core feature to one expression per family, with its unit and caveat.
The LycoSTand byte counts are payload only, so IP bytes are rebuilt as payload + L4 header + 20 bytes per
packet. Packet lengths are rebuilt with minimal IP+TCP/UDP headers.

`scripts/bridge_check.py` compares LycoS18 and NF-CSE-CIC-IDS2018-v2, which were extracted from the same PCAPs,
on 400k-row samples. Rule: a feature is dropped when its **benign** KS > 0.3.

| Feature | Benign KS | Decision |
|---|---|---|
| duration_s | 0.91 | **dropped**. NF-v2 duration is 0 for 82–94% of flows in every NF-v2 dataset (68% of NF-UNSW flows with >10 packets). A broken v2 field. |
| init_win_fwd / init_win_bwd | 0.38 / 0.45 | **dropped**. Initial window (LycoSTand) and maximum window (NetFlow) are different quantities. |
| the other 15 | 0.02–0.125 | kept. The rebuilt IP-byte and packet-length features agree (KS ≤ 0.11). |

Finding for H8: **within attack families the same traffic gives very different flows.** The median KS of packet
and byte counts per family is 0.78–0.98. LycoS18 has 1.80M DoS Hulk flows, while NF-CSE-CIC-IDS2018-v2 has 0.43M
from the same PCAPs. The extractors split attack traffic into flows differently, so an attack's "signature" is
partly an extractor artefact. Full table: `reports/tables/bridge_check.csv`.

### Label families (C2)

`configs/labels.yaml` maps every raw label seen in C1 to one of 8 families. Unmapped labels raise an error.
Judgement calls: NF-UNSW `Analysis` → Recon; NF-CSE `Brute Force -Web` → WebAttack (web-login brute force), not
BruteForce; ToN `password` → BruteForce; ToN `xss` and `injection` → WebAttack; Heartbleed, Infiltration,
Exploits, Fuzzers, Generic, Shellcode, Worms, mitm and ransomware → Other. Counts per (dataset, track, family)
are in `reports/tables/family_counts.csv`.

## Cleaning and splits (C3)

Configured in `configs/split.yaml` (version 1, seed 0, 60/20/20). The split files are `data/splits/{dataset}.parquet` with
(row_id, split, split_hash). Their hashes and sha256 are recorded in `reports/tables/splits_lock.csv`, which is committed.
**Frozen from 9 Oct 2026:** `scripts/split.py` refuses to overwrite a split whose hash changed unless `version` is
bumped, and a bump means re-running everything downstream. `make check-splits` verifies the files against the lock.
Two independent runs produced byte-identical split files: all randomness is splitmix64 on row ids, with no
dependence on the machine or the polars version.

**Dedup** is judged on each dataset's native track (cic77 / nf43 / cic_orig). Of each group of identical
(features, label) rows, one copy is kept. Groups whose identical features carry different labels are dropped
entirely. Equality uses a 64-bit hash of the feature vector. On nf_unsw_v2 and lycos17 an exact full-column
group-by reproduced the counts exactly.

| Dataset | Rows in | Exact duplicates | Conflict rows (vectors) | Kept | Train / val / test |
|---|---|---|---|---|---|
| lycos17 | 1,837,498 | 203,963 | 320 (16) | 1,633,215 | 978,127 / 327,115 / 327,973 |
| lycos18 | 13,691,268 | 2,926,527 | 296,512 (52) | 10,468,229 | 6,281,030 / 2,093,520 / 2,093,679 |
| nf_unsw_v2 | 2,390,275 | 1,634,749 | 28,786 (1,259) | 726,740 | 436,301 / 145,269 / 145,170 |
| nf_cse_cic18_v2 | 18,893,708 | 10,203,250 | 6,683,522 (14,056) | 2,006,936 | 1,204,068 / 400,861 / 402,007 |
| nf_ton_v2 | 16,940,496 | 13,127,092 | 1,965,035 (4,984) | 1,848,369 | 1,108,987 / 369,160 / 370,222 |
| cic17_orig | 2,830,743 | 325,496 | 7,144 (719) | 2,498,103 | 1,499,040 / 499,670 / 499,393 |

How to read the conflict column: exact dedup would have collapsed each conflict group to one row per label anyway.
So the real cost of the conflict rule is the number of **vectors** in brackets, not the row count. In NF-CSE,
6.67M of the 6.68M conflict rows sit in groups where one label holds ≥90%. These are typically Benign plus a few
`Infilteration` copies of the same vector (e.g. 66,331 Benign vs 469 Infilteration), i.e. label noise. In LycoS18,
identical vectors carry both FTP-Patator and DoS Slowhttptest labels, i.e. genuine ambiguity.

**Dedup changes the NetFlow datasets a lot.** NF-v2 flows are coarse integer records, so 80–90% of rows repeat
exactly. For example NF-CSE Bot falls from 143,097 rows to 472 unique vectors, and DDoS from 1.39M to 112,698.
Per-split family counts: `reports/tables/split_family_counts.csv`. Repeated vectors in the 15-feature `core`
track are not removed: dedup is defined on the native track, and two flows that differ natively are different
flows.

**Schemes.**
- lycos17: 5-minute time blocks, whole blocks assigned greedily, rarest family first, to the split furthest below
  its share of that family. Every family with ≥30 rows is in all three splits. Family proportions are coarser
  than 60/20/20 because attacks span few blocks (e.g. Recon 69/22/9%, DDoS 58/26/17%; DDoS covers only 5 blocks).
- All others: per-row seeded hash, which is stratified by family in expectation, plus a top-up so that every
  family with ≥30 rows appears in every split.
- When a dataset is a target, its train split is the unlabelled adaptation pool and its test split is for
  evaluation only.

**Run-level cleaning (C5)** is fit on source train only and stored in the model bundle:
- inf → NaN → source-train median (`MedianImputer`);
- dropping constant columns (`constant_cols`).

The dev subsample (`dev_subsample`) keeps at most 2M rows per split in family proportion, but never fewer than
5,000 rows of a family (or the whole family if smaller).

**Negative values are kept as values.**
- The `-1` sentinel ("not applicable", e.g. the TCP initial window of a non-TCP flow) appears in LycoS17 (~0.9M
  rows), LycoS18 (~4–5M) and cic17_orig.
- These are not sentinels but known extractor bugs, also kept: LycoS17 has 2,270 negative `iat_min` values (down
  to −14). cic17_orig has negative durations, rates and header lengths (down to −3.2e10).

## Demo data (C13): NFStream flows from raw PCAPs

- **nfs17** comes from the CIC-IDS2017 Monday, Wednesday and Friday PCAPs (UNB; `data/raw/cic17_pcap/`).
  - Flows are extracted with the frozen `configs/nfstream.yaml`, hash in `configs/nfstream.lock`. The extractor
    adds a TCP-termination plugin so that one connection is one flow, as in CICFlowMeter and LycoSTand.
  - Labels come from the LycoS17 rules (`xnids.data.labellers.lycos17`: epoch µs, exact attacker/victim pair).
  - 1,556,428 flows: Benign 1,100,862, DoS 201,033, PortScan 159,275, DDoS 94,522, Bot 735, Heartbleed 1.
  - Time-block split. The window 2017-07-05 13:30–13:55 UTC is held out of train/val/test because it is replay
    segment A.
- **nfs18** comes from CSE-CIC-IDS2018 Friday-16-02-2018 (AWS bucket, `scripts/fetch_cse18_pcaps.py`).
  - Contents: the victim web server 172.31.69.25 (2 parts) plus 13 of the 152 workstation captures (the largest
    9 capPC1 and 4 capDESKTOP hosts). SHA-256 values are in `data/MANIFEST.csv`.
  - Labels come from the official schedule, shifted by the measured clock offset: local = UTC−4.
    - DoS-SlowHTTPTest: 13.59.126.31 → victim, 10:12–11:08.
    - DoS-Hulk: 18.219.193.20 → victim, 13:45–14:19.
  - A flow seen by two capturing hosts (same 5-tuple and start time) is kept once.
  - 2,089,815 flows: DoS-Hulk 1,808,972, DoS-SlowHTTPTest 105,550, Benign 175,293. Stratified split.
- **Caveats:**
  - Every 2018 SlowHTTPTest flow is a refused connection to port 21, not slow HTTP.
  - The victim capture holds only Hulk's first 13 minutes; part 1 ends at 17:58 UTC.
  - capDESKTOP 66.85, 66.100 and 65.29 are corrupt partway through, upstream.
  - NFStream reports times in whole milliseconds, so exact duplicates are common: nfs17 dedup removes 44%.
- **Replay** (`scripts/build_demo_pcap.py`, `data/replay/`):
  - `demo.pcap` = segment A (2017, held out) + B1 + B2 (2018, time-shifted).
  - `demo_labels.parquet` holds one label per flow, keyed by 5-tuple and start time (`xnids.live.replay_labels`).

## Known limitations and confounds

- **cic17_orig is split at random, LycoS17 by time blocks.** This adds a split-scheme difference to the H2 confound. The original CIC timestamps are minute-resolution 12-hour strings with no AM/PM, so they cannot give reliable 5-minute blocks.
- **No timestamps in LycoS18 or NF-v2.** C3 can apply the time-block split only to LycoS17. Every other dataset gets a stratified random split, so near-duplicate neighbouring flows can leak across splits there. Deduplication before splitting reduces, but does not remove, this risk. NetFlow v3 (with timestamps) is the optional upgrade for the NF side.
- **Benign share varies widely:** 96% in NF-UNSW down to 36% in NF-ToN (an attack-majority dataset). This changes
  base rates across targets. FPR and DR are per-class rates and do not depend on the base rate; PR-AUC and MCC do.
- **H2 confound:** `cic17_orig` uses CICFlowMeter features and `lycos17` uses LycoSTand. Any difference therefore reflects both the extractor and the labels.

## Row counts (from `scripts/ingest.py`)

| Name | Rows after ingest | Published | Benign share | Notes |
|---|---|---|---|---|
| `lycos17` | 1,837,498 | n/a (matches authors' log) | 75.96% | 0 NaN / 0 inf |
| `lycos18` | 13,691,268 | 13,691,268 | 73.04% | All 14 class counts match the published README exactly. 0 NaN / 0 inf. |
| `nf_unsw_v2` | 2,390,275 | 2,390,275 | 96.02% | Upstream SHA-1 manifest verified. |
| `nf_cse_cic18_v2` | 18,893,708 | 18,893,708 | 88.05% | Upstream SHA-1 manifest verified. |
| `nf_ton_v2` | 16,940,496 | 16,940,496 | 36.01% | Upstream SHA-1 manifest verified. Attack-majority dataset. |
| `cic17_orig` | 2,830,743 | 2,830,743 | 80.30% | Raw CSVs hold 3,119,345 lines. The 288,602 extra are completely empty rows, dropped at ingest. `flow_bytes_s` has 1,358 NaN + 1,509 inf; `flow_packets_s` has 2,867 inf. Web-attack labels contain a mis-encoded dash (`Web Attack � XSS`), which is normalised in the C2 label map. |

Per-column details are in `reports/tables/profile_<name>.csv`, and label counts in `labels_<name>.csv`.
