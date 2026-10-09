# Model card: XGBoost on NFStream flows (`xgb-nfs-nfs17-s0-…`)

Base model of the `xgb:fewshot(...)` adaptation action (C14). Same data, features and threshold rule as the demo MLP.

- Within nfs17: FPR 0.86% at DR 92%, MCC 0.93. On nfs18: DR 0%.
- Few-shot (source train + labelled target flows upweighted to 20%): the only action that passes every gate in segment B. Results on segment B (`c14_actions_in_B.csv`): xgb:fewshot(budget=200,rule=random): B1 FPR 1.50%; xgb:fewshot(budget=1000,rule=random): B1 FPR 0.54%.
- Failure case: with 200 labels and target weight 0.2, about 1 pool in 41 yields a candidate that fails the canary-FPR gate (C16); the dashboard's Reject & rebuild draws a new label sample.
- With fewer than 5 attacks in the labelled budget the threshold is re-picked on source validation (fixed in C14: keeping the frozen threshold made a retrained model alert on every flow).
