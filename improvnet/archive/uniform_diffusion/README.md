# Archived uniform discrete diffusion

Legacy event-token uniform diffusion, retained for reproducibility on the
hierarchical-roll branch. Model/config/training, inference/evaluation, data/noise
helpers, benchmark, tests and documentation are grouped here. Shared AR code and
metric models remain in their original locations. Saved checkpoints and scratch
artifacts are not moved or changed.

Internal imports now use `improvnet.archive.uniform_diffusion`. Old module names
under `improvnet.model` and `improvnet.inference` are no longer entry points.

```bash
# Archived tests, opt-in rather than part of the active test directory.
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover \
  -s improvnet/archive/uniform_diffusion/tests

# Inspect the archived trainer; this does not start training.
python -m improvnet.archive.uniform_diffusion.model.uniform_diffusion_train --help
```

See [architecture](docs/uniform_diffusion_architecture.md),
[training](docs/uniform_diffusion_training.md), and
[evaluation](docs/uniform_diffusion_evaluation.md).
External SLURM launchers are not modified; their old module/file paths apply to
the old branch, not this archive. Use the new module paths if running legacy code
deliberately. The current model work is the local roll VAE followed by
hierarchical continuous diffusion.
