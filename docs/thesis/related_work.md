# Related work

The trading result is not a claim that a published network already makes money on this contract. The method is assembled from four groups of work, and the PO3/IFVG overlay is a hypothesis those papers do not test.

## Reward and optimizer

Moody and Saffell (NeurIPS 1998) and Moody, Wu, Liao, and Saffell (Journal of Forecasting, 1998) train a trader by ascending a differential Sharpe ratio instead of a supervised price forecast. A profit objective and a Sharpe objective learn different positions. That is the split in this thesis: Idea 3 uses differential Sharpe (`DSRReward`), and Ideas 1 and 2 use trade PnL. The optimizer is proximal policy optimization (Schulman, Wolski, Dhariwal, Radford, and Klimov, 2017), via Stable-Baselines3. PPO is the algorithm, not a published Nasdaq result.

## Trading literature

Zhang, Zohren, and Roberts (Journal of Financial Data Science, 2020) learn DQN, REINFORCE, and A2C policies on 50 futures from 2011 to 2019. Actions are `{-1, 0, 1}` or a continuous fraction in `[-1, 1]`, positions are scaled by target volatility over trailing volatility, and the non-learned control is the sign of a past return (Moskowitz, Ooi, and Pedersen, Journal of Financial Economics, 2012). Idea 3 is that discrete family with PPO in place of A2C, on one cash index, minute bars, a session clock, and structural stops. Their table cannot be copied: there is no official code, and the data are not this series.

FinRL (Liu, Yang, Gao, and Wang, ICAIF 2021) and the ensemble stock-trading study (Yang, Liu, Zhong, and Walid, ICAIF 2020) are why PPO is a reasonable default among A2C, DDPG, and SAC. Their agents are daily Dow 30 models. Their weights do not step this environment. The design already shared with that line of work is the calendar split: fit on the past, report on a later window that was not used to pick the model.

Jiang, Xu, and Liang (2017) map a price-history tensor through a CNN to portfolio weights. Bai, Kolter, and Koltun (2018) show a temporal convolution can match a recurrent net on sequence tasks, which is why TCN is compared with the GRU of Cho et al. (2014) and the Transformer of Vaswani et al. (2017). None of those papers is an intraday execution baseline.

## Holdout, selection, and the kill switch

Bailey, Borwein, López de Prado, and Zhu (Journal of Computational Finance, 2017) define the probability of backtest overfitting for the procedure that keeps the best of several trials. Bailey and López de Prado (Journal of Portfolio Management, 2014) deflate a Sharpe ratio by the number of trials and by skew and kurtosis. Six architectures and two strategy modes are one family of trials. López de Prado (Advances in Financial Machine Learning, 2018) is the source of the purged walk-forward inside the training calendar. The 2026 slice stays locked.

Achiam, Held, Tamar, and Abbeel (ICML 2017) and Ray, Achiam, and Amodei (2019) enforce risk as a constraint inside the policy update. This thesis enforces the daily-loss, max-loss, and trailing-drawdown limits in the environment and stops the episode. That is a kill switch, not constrained policy optimization. Huang and Ontañón (FLAIRS 2022) show that masking illegal actions is a valid policy gradient. Ideas 1 and 2 are closer to the options framework of Sutton, Precup, and Singh (Artificial Intelligence, 1999): the action sets a stop, a risk fraction, and a reward-to-risk ratio inside a hand-specified strategy, rather than a free long or short.

## What the overlay is not

Lo, Mamaysky, and Wang (Journal of Finance, 2000) find that some chart patterns are detectable in US stocks. They do not study power-of-three or inverse fair value gaps. There is no peer-reviewed performance result for those rules. Idea 2's "distribution" is a price-delivery phase, not distributional reinforcement learning. Buy-and-hold is one long from the first traded bar of the evaluation window. Its lot is fixed on the training path so that the worst price decline plus the overnight swap reaches the $10,000 max loss and stays inside the $5,000 daily loss. That same lot is then marked on the test window, with Wednesday swap counted three times and overnight margin at 1:15. Flat cash is a zero return. EMA/MACD/RSI and the sign of the previous session's return are separate active rules.
