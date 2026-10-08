# DriftGuard live demo (5 minutes)

**Setup** (once, about 1 minute before presenting):

```bash
make demo          # fresh detector API (data/live/demo) + dashboard on http://localhost:8501
```

Open http://localhost:8501. The sidebar should show *Active: mlp (original)*, *Candidate: —*, and *Replay: not started*.
Keep the default rates: **segment A 2000 flows/s, segment B 300 flows/s**. Every `make demo` starts from a clean
state: the original model and an empty audit log.

| Time | Do | Say / point at |
|---|---|---|
| 0:00 | Sidebar → **▶ Start replay**. Stay on **📈 Live**. | Real CIC-IDS2017 traffic (held-out minutes the model never saw) is going through NFStream into the detector. False positives stay around 1%; DoS-Hulk is caught at 94–98%. |
| ~1:30 | The dashed line on the charts is reached. | The replay switches to a different network: CSE-CIC-IDS2018, on AWS. Same tools, same attack names. Within seconds the original model's false-positive rate on benign traffic climbs to 7–11%. |
| ~2:00 | Open **🌊 Drift**. The banner turns red: **Drift alert**. | The monitor needs no labels. It counts flows that don't resemble anything the model was trained on, against a threshold calibrated on time-ordered 2017 traffic. It alerts after 2 windows over the threshold, so ordinary attack bursts don't trigger it. Read the plain-language sentences and point at the top shifted features. |
| ~2:20 | Open **🛠️ Adapt**. | The selector, trained on 6,500 past adaptations, recommends **xgb:fewshot(budget=200)**, with a predicted gain ± uncertainty. Label-free adapters would make things worse on this pair (C14). |
| ~2:30 | **Build candidate**. The gate table appears. | The candidate was trained on the 10,000 most recent flows plus 200 labelled ones. Here the replay's ground truth stands in for an analyst. Nothing is deployed yet. Four checks: canary detection rate, canary false-positive rate, parameter change, predicted attack rate. |
| (if a gate fails) | **↻ Reject & rebuild with a new label sample** | That is the gate doing its job: this sample of 200 labels produced a noisy model, which happens in about 2% of samples. Rebuild with a different sample. |
| ~3:00 | Type your name and a reason → **✅ Approve & promote**. Go back to **📈 Live**. | A human approves; the action is logged. The *adapted* line drops to about 0.5–1.5% FPR while the *original* (still scored as a shadow) stays at about 7%. |
| ~4:00 | **🛠️ Adapt** → name + reason → **↩ Roll back**. | One click restores the previous model. |
| ~4:30 | **📜 Audit**. | Every alert, recommendation, candidate, gate result, approval and rollback, with who did it and when. |

**Honest caveats** (on the dashboard as well):
- The FPR / DR lines are an *evaluation overlay*, available only because this is a replay of labelled traffic.
- The 2018 "DoS-SlowHTTPTest" flows are refused connections to port 21, not slow HTTP. After adapting on the new
  network's benign traffic, the model no longer flags them, and DoS-Hulk on the new network stays undetected (no
  Hulk flow from it was labelled).
- Alerts trail the packets by the flow-expiry lag shown in the sidebar: NFStream reports a flow when it ends.

**Rehearsal without a browser:** `make rehearse` runs this script 3 times through the real dashboard (Streamlit
AppTest) and checks every step (`reports/tables/c16_rehearsals.csv`).

**Backup recording:** record the screen while running the steps above (for example with OBS, or GNOME's
Ctrl+Alt+Shift+R) and keep the file next to the slides.
