# GPU step simulator (follow-on)

Not part of the six-run launch. The launch keeps `TradingEnv` and `SubprocVecEnv`.

## Why a GH200 stays partly idle

Each PPO step forwards 64 observations and then waits. Host RAM is about 270 GB per 64-env run because each worker is a Python process with its own environment. GPU memory stays near 5–15 GB. Sharing the observation matrix cuts the extra float32 copy. It does not turn the step into a GPU kernel.

## What the simulator would replace

The hot path in `quant_rl/envs/trading_env.py`: session-clipped observation window, account vector, fill, stop, and FTMO session reset. Features stay precomputed. The device holds one feature matrix plus a batched account state for every environment. One kernel advances all environments. The policy reads the batched observation without a pipe round-trip.

## Parity before any training use

Match `TradingEnv` on the same bars for:

- observation `seq` and `seq_mask` at a session open and after an overnight skip
- a forced close on the last bar of a session when `block_overnight` is set
- a max-loss breach ending the episode in train mode

Until those match, training stays on the CPU environment.
