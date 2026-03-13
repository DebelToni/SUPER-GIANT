# GIANT

`GIANT/` contains the versioned iterations of the main language model codebase.

## Version overview

- `v0` and `v1` are really old and not recommended for use. They serve as good learning resource because are simpler and mimic the early days of GPT-1/2/3
- `v2` is the stable base and the current reference implementation
- `v3` builds on top of `v2` and is the active branch for training-system improvements like multi-gpu training and more complex additions to the transformer like MLA.

## Future work

The next big steps planned for `v3` are:

- More advanced sharding beyond plain data parallelism
- Better partitioning of model and optimizer state
- More mature distributed execution strategies
- MLA-related experiments once the rest of the training stack is stable enough

If you want the stable path, start with `GIANT/v2`.
If you want the newest multi-GPU work, look at `GIANT/v3`.
