# Sharded bundle vector environment

`env.vector_backend: sharded_bundle` runs multiple logical environments inside
each subprocess. `env.n_envs` remains the logical PPO environment count;
`env.vector_shards` controls the number of worker processes. The worker groups
are contiguous and balanced, and each environment retains its own state, reset
seed, episode boundary, and terminal observation.

The backend is opt-in and requires `env.data_source: bundle`. Existing
`subproc` and `serial` backends remain unchanged. A behavior comparison must
hold source, bundle, seed, resolved PPO configuration, and logical `n_envs`
fixed. The throughput report records both the physical worker count and the
logical environments assigned to each worker.

Example profile override for an existing bundle:

```bash
python scripts/bench/measure_ppo_throughput.py \
  --bundle-dir /path/to/bundle \
  --overrides env.vector_backend=sharded_bundle env.vector_shards=8 \
  --output outputs/bench/sharded_bundle.json
```

This is an execution backend change only. It does not establish that the
backend is faster, improve policy quality, or change any historical campaign
result. Promote it only after exact seeded trace parity and a matched hardware
benchmark show an aggregate throughput benefit.
