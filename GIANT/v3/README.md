# GIANT v3

GIANT v3 builds directly on top of GIANT v2.

GIANT v2 is the stable base for the current data pipeline, model code, and training workflow. GIANT v3 keeps that foundation but is the place where I am modernizing the training stack and pushing distributed training further.

## Current focus

- Keep the v2-style training pipeline but make it cleaner and easier to scale
- Maintain strong single-GPU performance
- Provide working multi-GPU training with data parallelism
- Improve the dataset curation and curriculum setup

## Multi-GPU status

The multi-GPU path in this folder currently works in a DP-style setup.
Each GPU keeps a full model replica and gradients are synchronized across devices during training.

That is the most mature distributed setup in `GIANT/v3` today.

## Future direction

Once the current DP path is stable enough, this folder is where I want to explore:

- More advanced sharding strategies
- Better model and optimizer state partitioning
- More complex distributed layouts than plain DP
- MLA-style experiments in a cleaner and more mature training stack

So the short version is:

- `GIANT/v2` is the stable base
- `GIANT/v3` is the branch where the training system is being pushed forward
