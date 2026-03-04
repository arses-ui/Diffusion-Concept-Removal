# Shared Evaluation Protocol

This folder standardizes experimental runs across:
- `cure` (SD1.4 base CURE)
- `cure_seq` (sequential orthogonalized CURE)
- `cure_dit` (SD3/MM-DiT port)

## Files

- `protocol.py`: Shared concept sets, alpha rules, config helpers, JSON writer.
- `run_shared_eval.py`: Unified runner that generates before/after samples, applies erasure, and writes one result schema.

## Result Schema

Each run writes:
- `run_config.json`: Full run configuration snapshot.
- `results.partial.json`: Incremental checkpoint while running.
- `results.json`: Final output.

Top-level fields in `results.json`:
- `schema_version`
- `created_at_utc`
- `run_id`
- `git_commit`
- `config`
- `environment`
- `concept_results`
- `summary`

Each `concept_results[*]` entry includes:
- concept metadata (`concept`, `alpha`, prompt counts)
- `before`: image paths + generation time
- `erase`: erase time + raw eraser stats
- `after`: image paths + generation time

## Example Commands

```bash
# Base CURE, isolated per-concept runs (default for cure)
python evaluation/run_shared_eval.py \
  --branch cure \
  --concept-set objects10 \
  --embedding-mode mean_masked \
  --samples-per-concept 2

# CURE-Sequential, accumulated sequential erasures
python evaluation/run_shared_eval.py \
  --branch cure_seq \
  --concept-set objects10 \
  --erasure-mode sequential \
  --embedding-mode mean_masked

# CURE-DiT on SD3 (requires model access)
python evaluation/run_shared_eval.py \
  --branch cure_dit \
  --concept-set objects10 \
  --model-id stabilityai/stable-diffusion-3.5-medium \
  --embedding-mode mean_masked
```

## Notes

- `mean_masked` is the default embedding aggregation mode.
- Use `--embedding-mode token_flat` only for ablation.
- `cure_seq` supports only `sequential` mode.
