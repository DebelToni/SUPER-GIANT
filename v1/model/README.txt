This archive contains patch files that fix the four quality‑regression bugs:

1. **Unique layer scopes** – each `TinyTransformerBlock` now has its own
   `layer_{i}` param subtree so layers no longer collapse into one.

2. **Correct dropout RNG plumbing** – `Training_step.py` passes
   `rngs={"dropout": key}` so dropout really happens and gets a fresh key every
   batch.

3. **Model‑inside‑JIT removal** – parameters are created outside of any
   compiled graph; the `transformer_block_apply_jitted` helper **requires** an
   existing param subtree, and GiantGPT falls back to the slow path during the
   very first `model.init()`.

4. **Proper weight‑decay masking** – because every layer owns its own param
   path the original Optax mask once again works as intended.

Apply all three patches with:

    cd SUPER-GIANT/v1/model
    patch -p1 < Transformer_block.patch
    patch -p1 < GiantGPT.patch
    patch -p1 < Training_step.patch

or open each file and copy the changes manually.  After patching, re‑train:

    python Run_training.py

Quality should match the pre‑JIT run while keeping the speed‑ups.
