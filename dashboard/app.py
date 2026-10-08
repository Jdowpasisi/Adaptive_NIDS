"""C16 DriftGuard dashboard (Streamlit). Polls the detector API and its live.db every 2 s.

    make demo        # fresh API + this dashboard (scripts/run_demo.py)
    make dashboard   # this dashboard only, against $DG_API (default http://127.0.0.1:8000)

Pages: Live (original vs adapted, side by side), Drift (novelty, shifted features, MMD, ADWIN, explanation),
Adapt (recommendation, candidate, gate checks, Approve / Reject / Rollback -> API), Audit (who did what, when).
"""

import datetime as dt
import json
import os

import altair as alt
import polars as pl
import streamlit as st

from xnids.live.dashboard_data import DashboardData
from xnids.utils import paths

st.set_page_config(page_title="DriftGuard", page_icon="🛡️", layout="wide")

ORIG, ADAPT = "#eb6834", "#2a78d6"
COLORS = alt.Scale(domain=["original", "adapted"], range=[ORIG, ADAPT])
OVERLAY_NOTE = "Evaluation overlay: labels are known only because this is a replay of labelled traffic."


@st.cache_resource
def data_for(api: str) -> DashboardData:
    return DashboardData(api)


def short(v: str | None) -> str:
    if not v:
        return "—"
    base, _, rest = v.partition("+")
    return f"{base.split('-')[0]} + {rest.split('@')[0]}" if rest else base.split("-")[0] + " (original)"


def fmt(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def ts(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")


api = os.environ.get("DG_API", "http://127.0.0.1:8000")
D = data_for(api)

# ---------------------------------------------------------------- sidebar: status + replay control
with st.sidebar:
    st.title("🛡️ DriftGuard")
    h = D.health()
    if not h:
        st.error(f"Detector API not reachable at {api}")
        st.stop()
    st.caption(f"API {api}")
    st.markdown("**Replay control**")
    c1, c2 = st.columns(2)
    rate_a = c1.number_input("segment A flows/s", 100, 20000, 2000, 100, key="rate_a")
    rate_b = c2.number_input("segment B flows/s", 50, 20000, 300, 50, key="rate_b")
    b1, b2 = st.columns(2)
    if b1.button("▶ Start replay", key="start_replay", disabled=D.replay_running(), width="stretch"):
        D.start_replay(rate_a, rate_b)
        st.toast("Replay started: demo.pcap -> NFStream -> API")
    if b2.button("■ Stop", key="stop_replay", disabled=not D.replay_running(), width="stretch"):
        D.stop_replay()


@st.fragment(run_every=2)
def sidebar_status() -> None:
    with st.sidebar:
        m = D.get("/models") or {}
        st.markdown("**Models**")
        st.markdown(f"- Active: `{short((m.get('active') or {}).get('version'))}`\n"
                    f"- Candidate: `{short((m.get('candidate') or {}).get('version'))}`\n"
                    f"- Original (shadow): `{short(m.get('original'))}`")
        s = D.replay_status()
        st.markdown("**Replay**")
        if s:
            st.markdown(f"{'🟢 running' if s['running'] and not s['done'] else '⚪ finished'} · "
                        f"{s['flows_sent']:,} flows sent · {s['flows_per_s_10s']:.0f} flows/s")
            if s.get("lag_p50_s") is not None:
                st.caption(f"Flow-expiry lag (capture time): median {s['lag_p50_s']:.1f} s, p95 "
                           f"{s['lag_p95_s']:.0f} s. NFStream reports a flow when it ends (FIN/RST) or after "
                           "120 s idle, so alerts trail the packets.")
        else:
            st.caption("not started")


sidebar_status()

live, drift, adapt, audit = st.tabs(["📈 Live", "🌊 Drift", "🛠️ Adapt", "📜 Audit"])


# ---------------------------------------------------------------- Live
@st.fragment(run_every=2)
def live_page() -> None:
    m = D.get("/metrics") or {}
    models = D.get("/models") or {}
    ov = D.overlay(models.get("original"))
    apm = m.get("alerts_per_min", {})
    orig_apm = apm.get("original") if models.get("active", {}).get("version") != models.get("original") \
        else apm.get("active")
    adapted_apm = (apm.get("active") if models.get("active", {}).get("version") != models.get("original")
                   else apm.get("candidate"))
    k = st.columns(5)
    k[0].metric("Flows / s (60 s)", f"{m.get('flows_per_s', 0):,.0f}")
    k[1].metric("Alerts / min: original", f"{orig_apm or 0:,.0f}")
    k[2].metric("Alerts / min: adapted", f"{adapted_apm:,.0f}" if adapted_apm else "—")
    lat = m.get("latency_ms", {})
    k[3].metric("Scoring p99 latency", f"{lat['p99']:.0f} ms" if lat.get("p99") else "—")
    k[4].metric("Flows scored", f"{int(ov.filter(pl.col('series') == 'original')['n'].sum()) if not ov.is_empty() else 0:,}")
    if ov.is_empty():
        st.info("Waiting for flows. Start the replay in the sidebar.")
        return
    try:
        counts = json.loads((paths.REPLAY / "demo.json").read_text())["counts"]
        switch = sum(c["len"] for c in counts if c["segment"] == "A")      # known because this is a replay
    except (OSError, KeyError):
        switch = None
    df = ov.select("series", "streamed", (pl.col("fpr") * 100).alias("FPR %"), (pl.col("dr") * 100).alias("DR %"),
                   "alerts").to_pandas()
    rule = alt.Chart(pl.DataFrame({"x": [switch]}).to_pandas()).mark_rule(strokeDash=[4, 4], color="#898781") \
        .encode(x="x:Q") if switch else None
    for col, title in (("FPR %", "False-positive rate on benign flows (%)"),
                       ("DR %", "Detection rate on attack flows (%)")):
        c = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("streamed:Q", title="flows streamed"), y=alt.Y(f"{col}:Q", title=col),
            color=alt.Color("series:N", scale=COLORS, title=None),
            tooltip=["series", "streamed", alt.Tooltip(f"{col}:Q", format=".2f")]).properties(height=230, title=title)
        st.altair_chart(c + rule if rule is not None else c, width="stretch")
    st.caption(f"{OVERLAY_NOTE} Per 5,000 streamed flows; points need ≥ 50 benign / attack flows. Dashed line: "
               "the replay switches to the CSE-CIC-IDS2018 network (segment B). "
               "Segment B's 'DoS-SlowHTTPTest' flows are refused connections to port 21, not slow HTTP.")


with live:
    live_page()


# ---------------------------------------------------------------- Drift
@st.fragment(run_every=2)
def drift_page() -> None:
    d = D.drift()
    if d.is_empty():
        st.info("No drift report yet: the monitor reports every 5,000 flows.")
        return
    last = d.row(-1, named=True)
    (st.error if last["alert"] else st.success)(
        f"{'⚠️ Drift alert' if last['alert'] else '✅ No drift alert'} (window {last['window']}): {last['message']}")
    for e in last["explanations"][:5]:
        st.markdown(f"- {e}")
    thr = D.novelty_threshold()
    c1, c2 = st.columns([3, 2])
    nd = d.select("window", (pl.col("novelty") * 100).alias("novel %"), "alert").to_pandas()
    bars = alt.Chart(nd).mark_bar().encode(
        x=alt.X("window:O", title="5,000-flow window"), y=alt.Y("novel %:Q", title="novel flows (%)"),
        color=alt.condition("datum.alert", alt.value(ORIG), alt.value(ADAPT)), tooltip=["window", "novel %"])
    layers = bars
    if thr:
        layers = bars + alt.Chart(pl.DataFrame({"t": [thr * 100]}).to_pandas()).mark_rule(color="#52514e").encode(
            y="t:Q")
    c1.altair_chart(layers.properties(height=260, title="Novel flows per window (orange = alert: 2 windows over "
                                                       "the calibrated threshold)"), width="stretch")
    ks = sorted(last["ks"].items(), key=lambda kv: -kv[1][0])[:10]
    kd = pl.DataFrame({"feature": [f.replace("_", " ") for f, _ in ks], "KS": [v[0] for _, v in ks]}).to_pandas()
    c2.altair_chart(alt.Chart(kd).mark_bar(color=ADAPT).encode(
        x=alt.X("KS:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("feature:N", sort="-x", title=None))
        .properties(height=260, title="Top shifted features, latest window (KS)"), width="stretch")
    md = d.select("window", "mmd_p", "adwin").to_pandas()
    line = alt.Chart(md).mark_line(point=True, color="#1baf7a").encode(
        x=alt.X("window:O"), y=alt.Y("mmd_p:Q", title="MMD p-value", scale=alt.Scale(domain=[0, 1])))
    marks = alt.Chart(md[md.adwin]).mark_rule(color=ORIG, strokeDash=[3, 3]).encode(x="window:O")
    st.altair_chart((line + marks).properties(height=180, title="MMD p-value per window (dashed: ADWIN change on "
                                                                 "model confidence)"), width="stretch")
    st.caption("The alert rule is the C15 calibrated monitor: the share of flows unlike any reference 2017 flow, over "
               "its threshold for 2 consecutive windows. KS / MMD / ADWIN are shown for context: on real "
               "time-ordered traffic they also fire on ordinary attack bursts (C14 finding).")


with drift:
    drift_page()


# ---------------------------------------------------------------- Adapt
@st.fragment(run_every=2)
def adapt_page() -> None:
    h = D.health() or {}
    models = D.get("/models") or {}
    latest = D.get("/drift/latest") or {}
    if msg := st.session_state.pop("flash", None):               # set by the last action before its rerun
        st.success(msg)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Recommendation")
        if latest:
            act, gain, sd = latest.get("rec_action"), latest.get("rec_gain") or 0, latest.get("rec_std") or 0
            if act and act != "wait":
                st.markdown(f"**{act}**  \npredicted MCC gain **{gain:+.2f} ± {sd:.2f}** (selector, after label cost)")
            else:
                st.markdown("**wait**: no action is predicted to beat doing nothing (or no drift alert).")
        else:
            st.markdown("No drift report yet.")
        options = h.get("actions", [])
        rec = latest.get("rec_action")
        # follow each NEW recommendation; an analyst's own choice stands until the recommendation changes
        if rec in options and st.session_state.get("_last_rec") != rec:
            st.session_state["adapt_action"] = rec
            st.session_state["_last_rec"] = rec
        action = st.selectbox("Action", options, key="adapt_action")
        if st.button("Build candidate", key="build_candidate", disabled=bool(models.get("candidate")),
                     help="Runs the adapter on the 10,000 most recent flows. It never promotes."):
            with st.spinner(f"Running {action} on the recent flows …"):
                ok, out = D.post("/adapt", {"action": action, "requested_by": "dashboard",
                                            "seed": st.session_state.get("adapt_seed", 0)})
            if ok:
                st.session_state["flash"] = f"Candidate built: {short(out.get('candidate'))}"
                st.rerun()
            st.error(out.get("detail"))
        if "fewshot" in action:
            st.caption("Few-shot labels: the replay's ground truth stands in for an analyst labelling the selected "
                       "flows (200 or 1,000).")
    with c2:
        st.subheader("Candidate")
        cand = models.get("candidate")
        if not cand:
            st.markdown("No candidate.")
        else:
            g = cand.get("gates", {})
            st.markdown(f"`{cand['version']}`  \nbuilt by **{cand.get('created_by')}** with **{cand.get('action')}** "
                        f"at {ts(cand['since'])}")
            rows = []
            for name, v in g.items():
                if not isinstance(v, dict):
                    continue
                rows.append({"check": name, "candidate": fmt(v.get("value")), "active": fmt(v.get("active")),
                             "limit": fmt(v.get("limit")),
                             "result": {True: "✅ pass", False: "❌ fail", None: "– n/a"}[v.get("passed")]})
            st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")
            if g.get("all_passed"):
                st.success("All gate checks passed: the candidate may be promoted.")
            else:
                st.error("A gate check failed: promotion is blocked.")
                if st.button("↻ Reject & rebuild with a new label sample", key="rebuild",
                             help="Rejects this candidate and runs the same action again with the next random "
                                  "seed: a different set of labelled flows."):
                    seed = st.session_state.get("adapt_seed", 0) + 1
                    st.session_state["adapt_seed"] = seed
                    D.post("/models/reject", {"requested_by": "dashboard", "reason": "gate check failed; rebuild"})
                    with st.spinner(f"Rebuilding {cand.get('action')} (seed {seed}) …"):
                        ok, out = D.post("/adapt", {"action": cand.get("action"), "requested_by": "dashboard",
                                                    "seed": seed})
                    st.session_state["flash"] = (f"Rebuilt with seed {seed}: {short(out.get('candidate'))}" if ok
                                                 else f"Rebuild failed: {out.get('detail')}")
                    st.rerun()
    st.divider()
    a, r = st.columns(2)
    with a:
        st.subheader("Approve / Reject")
        with st.form("decision", clear_on_submit=False):
            who = st.text_input("Your name", key="approver")
            why = st.text_input("Reason", key="reason")
            ca, cr = st.columns(2)
            approve = ca.form_submit_button("✅ Approve & promote", disabled=not (models.get("candidate") or {})
                                            .get("gates", {}).get("all_passed"))
            reject = cr.form_submit_button("✖ Reject", disabled=not models.get("candidate"))
        if approve:
            ok, out = D.post("/models/promote", {"approved_by": who, "reason": why})
            if ok:
                st.session_state["flash"] = f"Promoted: {short(out.get('active'))}"
                st.rerun()
            st.error(f"Refused: {out.get('detail')}")
        if reject:
            ok, out = D.post("/models/reject", {"requested_by": who or "?", "reason": why or "?"})
            if ok:
                st.session_state["flash"] = "Candidate rejected"
                st.rerun()
            st.error(out.get("detail"))
    with r:
        st.subheader("Rollback")
        hist = models.get("history", [])
        st.markdown(f"Previous active: `{short(hist[-1])}`" if hist else "No previous version.")
        with st.form("rollback"):
            rwho = st.text_input("Your name", key="rb_name")
            rwhy = st.text_input("Reason", key="rb_reason")
            go = st.form_submit_button("↩ Roll back", disabled=not hist)
        if go:
            ok, out = D.post("/models/rollback", {"requested_by": rwho, "reason": rwhy})
            if ok:
                st.session_state["flash"] = f"Rolled back: {short(out.get('active'))} is active again"
                st.rerun()
            st.error(out.get("detail"))


with adapt:
    adapt_page()


# ---------------------------------------------------------------- Audit
@st.fragment(run_every=2)
def audit_page() -> None:
    a = D.get("/audit", limit=500) or []
    if not a:
        st.info("Audit log is empty.")
        return
    icon = {"drift_alert": "⚠️", "recommendation": "💡", "adapt": "🛠️", "gate_check": "🧪", "promote": "✅",
            "promote_refused": "⛔", "reject": "✖", "rollback": "↩", "service_start": "▶", "adapt_failed": "❗"}
    rows = [{"time": ts(x["ts"]), "event": f"{icon.get(x['kind'], '•')} {x['kind']}", "who": x["actor"],
             "from": short(x["version_from"]), "to": short(x["version_to"]), "reason": x["reason"]} for x in a]
    st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch", height=520)


with audit:
    audit_page()
