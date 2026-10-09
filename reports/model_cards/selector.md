# Model card: adapter selector (`models/selector/`)

Bootstrap ensemble of 5 LightGBM regressors predicting each action's MCC gain minus label cost from the label-free drift features of a window (C11).

- Training data: the C10 adaptation log (249a8e1f10), 6,500 rows over 8 dataset pairs; utility = mcc, label cost 0.0002 per label.
- Drift features: ks_max, ks_mean, n_ks_sig, mmd_stat, mmd_p, adwin_flag, mean_conf, entropy, est_fpr_rise, attack_share_pred.
- Leave-one-pair-out regret 0.077 vs 0.050 for the best fixed action (XGBoost few-shot 200): it does not beat the best fixed action on average (beats it on 3 of 8 held-out pairs).
- In the live system it chooses among the enabled actions (`configs/live.yaml`) and recommends XGBoost few-shot 200 on the new network, the action that works there.
- MLflow run: `be797c42afb3415ca7908554e0855591`.
