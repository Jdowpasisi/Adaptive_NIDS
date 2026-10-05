# Dataset card: DriftGuard benchmark

Status: v1 (C1 ingest + C2 tracks and labels). C3 adds cleaning counts and the split scheme.
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

## Known limitations and confounds

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
