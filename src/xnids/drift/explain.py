"""Plain-language explanations of the most shifted features, e.g. "flow duration is 3.1x longer than in training"."""

import numpy as np

TOKENS = {"pkt": "packet", "pkts": "packets", "fwd": "forward", "bwd": "backward", "iat": "inter-arrival time",
          "len": "length", "tot": "total", "cnt": "count", "hdr": "header", "s": "per second", "win": "window",
          "init": "initial", "prot": "protocol", "dst": "destination", "src": "source", "l4": "L4", "ttl": "TTL"}


def pretty(name: str) -> str:
    """Whole-word expansion: 'fwd_pkt_len_tot' -> 'forward packet length total'."""
    return " ".join(TOKENS.get(t, t) for t in name.split("_"))


def describe(ref: np.ndarray, win: np.ndarray, name: str, ks_stat: float) -> str:
    """One sentence for one feature, chosen by what actually changed: zero share, median ratio, or the shape."""
    ref, win = ref[np.isfinite(ref)], win[np.isfinite(win)]
    label = pretty(name)
    if len(ref) == 0 or len(win) == 0:
        return f"{label} shifted (KS {ks_stat:.2f})"
    z_r, z_w = float((ref == 0).mean()), float((win == 0).mean())
    if abs(z_w - z_r) >= 0.25:
        return f"{label} is zero in {100 * z_w:.0f}% of flows vs {100 * z_r:.0f}% in training"
    m_r, m_w = float(np.median(ref)), float(np.median(win))
    if m_r > 0 and m_w > 0:
        ratio = m_w / m_r
        if ratio >= 1.5:
            return f"{label} is {ratio:.1f}x higher than in training (median {m_w:.3g} vs {m_r:.3g})"
        if ratio <= 1 / 1.5:
            return f"{label} is {1 / ratio:.1f}x lower than in training (median {m_w:.3g} vs {m_r:.3g})"
    q = [0.1, 0.9]
    r_lo, r_hi = np.quantile(ref, q)
    w_lo, w_hi = np.quantile(win, q)
    return (f"{label} has a different distribution (KS {ks_stat:.2f}; middle 80% now {w_lo:.3g}-{w_hi:.3g}, "
            f"was {r_lo:.3g}-{r_hi:.3g})")


def message(sentences: list[str], drift: bool) -> str:
    if not drift:
        return "No significant drift in this window."
    return "Traffic has shifted: " + "; ".join(sentences[:3]) + "."
