from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import orbax.checkpoint as ocp
from flax.training import orbax_utils


@dataclass
class AsyncMiniCheckpointManager:
    ckpt_dir: Path
    max_to_keep: int = 3

    def __post_init__(self):
        self.ckpt_dir = Path(self.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        handler = ocp.PyTreeCheckpointHandler()
        self.checkpointer = ocp.AsyncCheckpointer(handler)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=self.max_to_keep,
            create=True,
        )
        self.manager = ocp.CheckpointManager(
            str(self.ckpt_dir),
            self.checkpointer,
            options,
        )

    def latest_step(self) -> int | None:
        return self.manager.latest_step()

    def restore_latest(self, target: Any) -> tuple[Any, int]:
        """
        Restore into `target` (PyTree template) from latest mini checkpoint.
        Returns (restored_target, step). If none exist, returns (target, 0).
        """
        step = self.manager.latest_step()
        if step is None:
            return target, 0
        restored = self.manager.restore(step, items=target)
        return restored, step

    def save(self, step: int, state: Mapping[str, Any]) -> None:
        """
        Asynchronously save `state` at `step`.
        Returns quickly; actual I/O is in background threads.
        """
        save_args = orbax_utils.save_args_from_target(state)
        self.manager.save(
            step,
            state,
            save_kwargs={"save_args": save_args},
        )

    def wait_until_finished(self, timeout: float | None = None) -> None:
        """
        Optionally wait for any pending async writes.
        """
        _ = timeout  # retained for API compatibility; AsyncCheckpointer does not take a timeout arg.
        self.checkpointer.wait_until_finished()
