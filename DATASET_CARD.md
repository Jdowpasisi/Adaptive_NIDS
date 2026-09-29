# Dataset card: DriftGuard benchmark

Status: v0 (C1 ingest). Later components add the tracks, label map, cleaning counts and split scheme.
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

## Known limitations and confounds

- **No timestamps in LycoS18 or NF-v2.** C3 can apply the time-block split only to LycoS17. Every other dataset gets a stratified random split, so near-duplicate neighbouring flows can leak across splits there. Deduplication before splitting reduces, but does not remove, this risk. NetFlow v3 (with timestamps) is the optional upgrade for the NF side.
- **H2 confound:** `cic17_orig` uses CICFlowMeter features and `lycos17` uses LycoSTand. Any difference therefore reflects both the extractor and the labels.

## Row counts (from `scripts/ingest.py`)

| Name | Rows after ingest | Published | Benign share | Notes |
|---|---|---|---|---|
| `lycos17` | 1,837,498 | n/a (matches authors' log) | 75.96% | 0 NaN / 0 inf |
| `lycos18` | 13,691,268 | 13,691,268 | 73.04% | All 14 class counts match the published README exactly. 0 NaN / 0 inf. |
| `cic17_orig` | 2,830,743 | 2,830,743 | 80.30% | Raw CSVs hold 3,119,345 lines. The 288,602 extra are completely empty rows, dropped at ingest. `flow_bytes_s` has 1,358 NaN + 1,509 inf; `flow_packets_s` has 2,867 inf. Web-attack labels contain a mis-encoded dash (`Web Attack � XSS`), which is normalised in the C2 label map. |

Per-column details are in `reports/tables/profile_<name>.csv`, and label counts in `labels_<name>.csv`.
