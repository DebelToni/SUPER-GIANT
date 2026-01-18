from __future__ import annotations

import numpy as np

from TiDAR.model.tidar_utils import build_train_batch


def main() -> None:
    tokens = np.arange(1, 17, dtype=np.int32)[None, :]
    lengths = np.array([16], dtype=np.int32)
    batch = build_train_batch(tokens, lengths, mask_id=999, block_len=4)

    input_ids = np.asarray(batch["input_ids"])[0]
    position_ids = np.asarray(batch["position_ids"])[0]
    labels = np.asarray(batch["labels"])[0]
    mask_ntp = np.asarray(batch["loss_mask_ntp"])[0]
    mask_diff = np.asarray(batch["loss_mask_diff"])[0]

    assert input_ids.shape[0] == 32
    assert position_ids[:16].tolist() == list(range(16))
    assert position_ids[16:].tolist() == list(range(16))

    assert labels[:15].tolist() == list(range(2, 17))
    assert labels[15] == -100
    assert labels[16:].tolist() == list(range(1, 17))

    assert mask_ntp[:15].sum() == 15
    assert mask_ntp[15:].sum() == 0
    assert mask_diff[:16].sum() == 0
    assert mask_diff[16:].sum() == 16

    print("TiDAR train batch looks correct")


if __name__ == "__main__":
    main()
