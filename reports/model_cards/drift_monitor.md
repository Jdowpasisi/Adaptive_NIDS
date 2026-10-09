# Model card: live drift monitor (`models/monitor/nfs-nfs17.json`)

Calibrated for time-ordered real traffic (C15).

- Statistic: share of a window's flows whose nearest reference flow (standardised MLP embedding) is farther than 99% of held-out source flows (flow cutoff 1.075).
- Window threshold: 3.200% novel flows, the 99th percentile over 268 time-ordered windows of raw (not deduplicated) 2017 traffic; alert after 2 consecutive windows over it.
- Demo replay: 0 of 35 segment-A windows alert; first alert at the 2nd window of the new network.
- Known failure: the C8 monitor (KS / MMD / ADWIN + cost trigger) false-alarmed on 34 of 35 in-distribution windows of real traffic; KS / MMD cannot tell an attack burst from a new network. ATC's estimated error rise was higher on in-distribution traffic than on the new network, so it is not used for alerting.
- Novelty in input space missed the evasive DoS-Hulk windows (20% over threshold vs 80% in embedding space).
