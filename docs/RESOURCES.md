# Compute plan and crash safety

Measured on the team laptop: 12 threads, 14.9 GB RAM, **no swap**, RTX 3050 4 GB. The project drive
(`/mnt/windows-d`, NTFS) has 104 GB free and the root partition (ext4) 30 GB free (as of 7 Oct 2026).

## Measured so far

| Job | Wall time | CPU time | Peak RAM | Notes |
|---|---|---|---|---|
| C1 ingest LycoS18 (13.7M rows) | ~3 min | — | 7.8 GB | streaming CSV → Parquet |
| C3 splits (all datasets) | ~10 min | — | 4.8 GB | |
| C4 adversarial validation (84 curves) | 3.5 h | — | < 4 GB | resumable |
| C6 sweep (25 configs) + TabNet redo + H2 control | 4.4 h + 0.5 h | 17 h | **11.3–11.9 GB** | RF and TabNet on LycoS18 (2M rows) dominate |
| C8 drift evaluation (8 pairs × 3 seeds) | ~25 min | — | see C8 report | GPU for MMD |

## Estimates for what remains (laptop, dev subsample ≤ 2M rows per split)

| Component | Estimate | Peak RAM | GPU | Main cost |
|---|---|---|---|---|
| C7 decomposition (ablation k ∈ {1,3,5,10} × RF/XGB/MLP, normalisation variant, time drift) | 5–6 h | ~11 GB | yes (MLP) | RF retraining on LycoS18 |
| C9 adapters (6 adapters, cic77 + ≥ 1 nf43 pair, 3 seeds) | 4–6 h | 8–11 GB | yes | CORAL/DANN training, few-shot tree retraining |
| C10 adaptation logs (≥ 2,000 rows) | 6–10 h | ~10 GB | yes | every action × ~400 windows |
| C11 selector | < 0.5 h | < 4 GB | no | |
| C12 Reptile (meta-training + leave-one-target-out) | 3–5 h | < 6 GB | yes | |
| C13 NFStream on 3 CIC-2017 PCAPs (33 GB) + one CSE-CIC-2018 day | 1–2 h CPU + download | 2–6 GB | no | plus a 10–20 GB PCAP download |
| C14–C16 API, replay, dashboard | development; load tests < 1 h | < 4 GB | no | |
| C17 poisoning (2 attacks × 3 ρ × guards × 3 seeds) | 2–3 h | < 8 GB | yes | |
| **Total remaining** | **~25–40 h of compute** | | | |

Disk: about +20–30 GB more (CSE-CIC-2018 PCAP, mlruns growth, model bundles). That fits on the project drive.

## What can go wrong and what protects us

| Risk | Likelihood | Protection |
|---|---|---|
| Out-of-memory kill (peaks reach 11–12 GB of 14.9, no swap) | **high** on RF / TabNet / LycoS18 jobs | One process per config. The TabNet importance pass is off. **Add swap (below).** Close the browser during big runs. |
| Session/terminal closes | happened twice | Long jobs run as `systemd-run --user` units and survive. Fully logging out stops them. |
| Laptop crash / power loss mid-run | medium | Every long job is resumable: finished (config hash, seed) runs are skipped, and an interrupted run is simply redone. Run on AC power. |
| `mlflow.db` (SQLite on NTFS) corrupted by a crash | low, but costly | `make backup` after every long job: a consistent snapshot to `~/driftguard_backup` (ext4); 7 snapshots kept. |
| Lost data files | low | `data/raw` is re-downloadable (checksums in `data/MANIFEST.csv`). processed/interim rebuild in ~1 h. Frozen splits are backed up and verified with `make check-splits`. |
| Laptop unavailable for days | — | Offload C7 / C10 to Kaggle (29 GB RAM, GPU): upload `data/processed` + `data/splits` (~3 GB), run with `max_rows_per_split: null`, copy the run folders back. |

## Recommended one-time setup (needs sudo)

An 8 GB swapfile on the ext4 root turns OOM kills into slowdowns. It fits next to the backups in the 30 GB free.

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap defaults 0 0' | sudo tee -a /etc/fstab
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swappiness.conf && sudo sysctl --system
```

## Routine

```bash
systemctl --user list-units 'dg-*'     # what is running
tail -f logs/<job>.log                 # progress
make backup                            # after every long job (safe while jobs run)
git push origin main                   # code, configs and reports/ are in git
```
