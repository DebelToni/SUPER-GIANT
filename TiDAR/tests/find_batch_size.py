#!/usr/bin/env python3
"""
Find the optimal (maximum) batch size for TiDAR training that doesn't OOM.

This script uses binary search to find the largest batch size that allows
training to run successfully for a few steps without running out of GPU memory.

Usage:
    python TiDAR/tests/find_batch_size.py --config TiDAR/model/model_configs/old/sweep_5_lr3e5_a0p2_l0p8_t2_ctxmix.yml

The script:
1. Parses the config and finds the stage with the longest context length
2. Creates a temporary test config using only that stage (worst-case memory)
3. Spawns training runs as separate processes to ensure clean GPU memory state
4. Uses binary search (O(log n)) to efficiently find the maximum batch size
5. Monitors the logs.txt file to determine if training succeeded
6. Detects JAX OOM errors by looking for specific error messages in stderr
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import yaml
from pathlib import Path

# OOM and memory error patterns that JAX/XLA produce
OOM_PATTERNS = [
    "out of memory",
    "Out of memory",
    "OOM",
    "RESOURCE_EXHAUSTED",
    "ResourceExhaustedError",
    "XLA:GPU ran out of memory",
    "Could not allocate",
    "failed to allocate",
    "Rematerialization failed",
    "Unable to allocate",
    "AllocationNotFound",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "cudaMalloc failed",
    "Failed to allocate memory",
]

# Patterns indicating JAX compilation issues (not OOM but might precede OOM)
COMPILATION_ISSUE_PATTERNS = [
    "Compilation failed",
    "ptxas error",
    "jaxlib.xla_extension.XlaRuntimeError",
]

# Patterns indicating config/setup errors (not OOM, test may be invalid)
CONFIG_ERROR_PATTERNS = [
    "requires positive decay_steps",
    "ConfigKeyError",
    "ConfigAttributeError",
    "FileNotFoundError",
    "Dataset directory",
    "missing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find optimal batch size for TiDAR training via binary search"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model config YAML file (same as Run_training.py --config)",
    )
    parser.add_argument(
        "--global_config",
        type=str,
        default=None,
        help="Optional path to global config (same as Run_training.py --global_config)",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Initial checkpoint to load (same as Run_training.py --init_checkpoint)",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=1,
        help="Minimum batch size to start searching from (default: 1)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=256,
        help="Maximum batch size to search up to (default: 256)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Log every N steps to detect successful training (default: 1)",
    )
    parser.add_argument(
        "--wait_for_logs",
        type=int,
        default=5,
        help="Number of log entries to wait for before declaring success (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each training run (default: 600)",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory for checkpoints (default: auto-create in /tmp)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python interpreter to use (default: same as current process)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output including subprocess stderr",
    )
    return parser.parse_args()


def load_config_yaml(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_max_seqlen_stage(config: dict) -> tuple[dict, int]:
    """Find the stage with the maximum sequence length.
    
    Returns:
        (stage_dict, seq_len): The stage with max seq_len and its seq_len value
    """
    stages = config.get("stages", [])
    if not stages:
        raise ValueError("Config has no stages defined")
    
    max_stage = max(stages, key=lambda s: s.get("seq_len", 0))
    max_seq_len = max_stage.get("seq_len", 2048)
    return max_stage, max_seq_len


def create_test_config(
    original_config: dict,
    test_stage: dict,
    temp_dir: Path,
) -> str:
    """Create a temporary config file for batch size testing.
    
    Modifies the config to:
    - Use only the specified stage (with longest seq_len)
    - Set end_ratio to 1.0 for that stage
    - Set warmup_steps to 10 (minimal warmup for testing)
    - Disable mini checkpoints
    """
    test_config = original_config.copy()
    
    # Create a modified stage for testing
    test_stage_copy = test_stage.copy()
    test_stage_copy["end_ratio"] = 1.0
    test_stage_copy["epochs"] = 1
    test_config["stages"] = [test_stage_copy]
    
    # Modify optimizer for faster testing
    if "optimizer" in test_config:
        test_config["optimizer"] = test_config["optimizer"].copy()
        test_config["optimizer"]["warmup_steps"] = 10
    
    # Modify training settings
    if "training" in test_config:
        test_config["training"] = test_config["training"].copy()
        test_config["training"]["mini_checkpoint_every"] = 999999
        test_config["training"]["checkpoint_every"] = 999999
        test_config["training"]["log_every"] = 1
    
    # Write to temp file
    test_config_path = temp_dir / "test_config.yml"
    with open(test_config_path, "w", encoding="utf-8") as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    return str(test_config_path)


def check_oom_in_output(output: str) -> bool:
    """Check if the output contains OOM-related error messages."""
    output_lower = output.lower()
    for pattern in OOM_PATTERNS:
        if pattern.lower() in output_lower:
            return True
    return False


def check_compilation_issues(output: str) -> bool:
    """Check if output has compilation issues (may precede OOM)."""
    for pattern in COMPILATION_ISSUE_PATTERNS:
        if pattern in output:
            return True
    return False


def check_config_errors(output: str) -> tuple[bool, str | None]:
    """Check if output has config/setup errors (not OOM, test invalid).
    
    Returns:
        (has_error, error_description): tuple of whether error found and optional description
    """
    for pattern in CONFIG_ERROR_PATTERNS:
        if pattern in output:
            return True, pattern
    return False, None


def count_log_entries(log_path: Path) -> int:
    """Count the number of log entries in logs.txt."""
    if not log_path.exists():
        return 0
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Count lines that look like actual log entries (contain "step")
        return sum(1 for line in lines if "step" in line.lower())
    except Exception:
        return 0


def get_log_file_mtime(log_path: Path) -> float | None:
    """Get modification time of log file, or None if it doesn't exist."""
    if not log_path.exists():
        return None
    try:
        return log_path.stat().st_mtime
    except Exception:
        return None


def run_training_test(
    config_path: str,
    batch_size: int,
    checkpoint_dir: Path,
    *,
    global_config: str | None = None,
    init_checkpoint: str | None = None,
    log_every: int = 1,
    wait_for_logs: int = 5,
    timeout: int = 600,
    python_path: str | None = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Run a training test with the given batch size.

    Returns:
        (success, message): success is True if training ran successfully for a few steps
    """
    python_exe = python_path or sys.executable
    script_path = Path(__file__).parent.parent / "model" / "Run_training.py"

    # Build command
    cmd = [
        python_exe,
        str(script_path),
        "--config", config_path,
        "--batch_size", str(batch_size),
        "--gradient_accumulation", "1",  # Always 1 for batch size testing
        "--log_every", str(log_every),
        "--checkpoint_dir", str(checkpoint_dir),
        "--checkpoint_every", "999999",  # Don't save actual checkpoints
    ]
    if global_config:
        cmd.extend(["--global_config", global_config])
    if init_checkpoint:
        cmd.extend(["--init_checkpoint", init_checkpoint])

    log_path = checkpoint_dir / "params" / "logs.txt"

    # Clean up any existing logs
    if log_path.exists():
        log_path.unlink()

    print(f"  Starting training with batch_size={batch_size}...")
    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    # Start the process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "JAX_TRACEBACK_FILTERING": "off"},
        )
    except Exception as e:
        return False, f"Failed to start process: {e}"

    # Monitor the process
    start_time = time.time()
    last_log_count = 0
    last_log_mtime = None
    logs_stable_time = None
    initial_log_created = False

    try:
        while True:
            # Check if process has terminated
            retcode = process.poll()
            elapsed = time.time() - start_time

            # Check for timeout
            if elapsed > timeout:
                process.kill()
                process.wait()
                return False, f"Timeout after {timeout}s"

            # Check log file
            current_log_count = count_log_entries(log_path)
            current_mtime = get_log_file_mtime(log_path)

            # Detect initial log creation
            if not initial_log_created and current_log_count > 0:
                initial_log_created = True
                print(f"    [+{elapsed:.1f}s] Log file created, first entry detected")

            # Track log changes after initial creation
            if initial_log_created:
                if current_log_count > last_log_count:
                    print(f"    [+{elapsed:.1f}s] Log entries: {current_log_count}")
                    last_log_count = current_log_count
                    last_log_mtime = current_mtime
                    logs_stable_time = None  # Reset stability timer

                # Check if we've seen enough log entries with continued writes
                if current_log_count >= wait_for_logs:
                    # Wait a bit more to see if it keeps writing (indicating stability)
                    if logs_stable_time is None:
                        logs_stable_time = time.time()
                    elif time.time() - logs_stable_time > 3.0:
                        # Logs have been stable for 3s with enough entries
                        process.kill()
                        process.wait()
                        return True, f"Success: {current_log_count} log entries after {elapsed:.1f}s"

            # If process exited, check why
            if retcode is not None:
                stdout, stderr = process.communicate()
                combined_output = stdout + stderr

                if check_oom_in_output(combined_output):
                    return False, "OOM detected in output"
                if check_compilation_issues(combined_output):
                    return False, "Compilation issues detected"
                
                # Check for config errors (test may be invalid at this batch size)
                has_config_error, config_pattern = check_config_errors(combined_output)
                if has_config_error:
                    return False, f"Config error ({config_pattern})"

                if verbose:
                    print(f"    Process exited with code {retcode}")
                    if stderr:
                        print(f"    stderr:\n{stderr[:2000]}")

                # If we got enough logs before exit, consider it a success
                if current_log_count >= wait_for_logs:
                    return True, f"Success (process exited): {current_log_count} log entries"

                return False, f"Process exited with code {retcode}, only {current_log_count} logs"

            # Short sleep before next check
            time.sleep(0.5)

    except KeyboardInterrupt:
        process.kill()
        process.wait()
        raise
    finally:
        # Make sure process is dead
        if process.poll() is None:
            process.kill()
            process.wait()


def binary_search_batch_size(
    config_path: str,
    min_bs: int,
    max_bs: int,
    temp_dir: Path,
    **kwargs,
) -> tuple[int, list[tuple[int, bool, str]]]:
    """
    Binary search to find the maximum working batch size.

    Returns:
        (max_working_batch_size, history): history is list of (batch_size, success, message)
    """
    history: list[tuple[int, bool, str]] = []
    best_working = 0
    low, high = min_bs, max_bs

    # First check if min_bs works at all
    print(f"\n[Phase 1] Testing minimum batch size {min_bs}...")
    success, msg = run_training_test(config_path, min_bs, temp_dir, **kwargs)
    history.append((min_bs, success, msg))
    print(f"  Result: {'SUCCESS' if success else 'FAILED'} - {msg}")

    if not success:
        print(f"\nERROR: Even minimum batch size {min_bs} failed. Cannot continue.")
        return 0, history

    best_working = min_bs

    # Check if max_bs works (then we're done quickly)
    print(f"\n[Phase 2] Testing maximum batch size {max_bs}...")
    success, msg = run_training_test(config_path, max_bs, temp_dir, **kwargs)
    history.append((max_bs, success, msg))
    print(f"  Result: {'SUCCESS' if success else 'FAILED'} - {msg}")

    if success:
        print(f"\nMaximum batch size {max_bs} works! No need for further search.")
        return max_bs, history

    # Binary search between low and high
    print(f"\n[Phase 3] Binary search between {min_bs} and {max_bs}...")
    low = min_bs
    high = max_bs

    while low < high - 1:
        mid = (low + high) // 2
        print(f"\n  Testing batch_size={mid} (range: [{low}, {high}])...")

        success, msg = run_training_test(config_path, mid, temp_dir, **kwargs)
        history.append((mid, success, msg))
        print(f"  Result: {'SUCCESS' if success else 'FAILED'} - {msg}")

        if success:
            best_working = mid
            low = mid
        else:
            high = mid

    # Final verification: test low+1 to high-1 if there's a gap
    if high - low > 1:
        for bs in range(low + 1, high):
            if bs not in [h[0] for h in history]:
                print(f"\n  Testing batch_size={bs} (final verification)...")
                success, msg = run_training_test(config_path, bs, temp_dir, **kwargs)
                history.append((bs, success, msg))
                print(f"  Result: {'SUCCESS' if success else 'FAILED'} - {msg}")
                if success:
                    best_working = max(best_working, bs)

    return best_working, history


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("TiDAR Batch Size Finder")
    print("=" * 60)
    print(f"Config: {args.config}")
    
    # Load and analyze config
    original_config = load_config_yaml(args.config)
    max_stage, max_seq_len = find_max_seqlen_stage(original_config)
    
    print(f"Testing with stage: {max_stage.get('name', 'unknown')}")
    print(f"  - dataset: {max_stage.get('dataset', 'unknown')}")
    print(f"  - seq_len: {max_seq_len} (maximum across all stages)")
    print(f"Search range: [{args.min_batch_size}, {args.max_batch_size}]")
    print(f"Wait for {args.wait_for_logs} log entries, timeout={args.timeout}s")
    print("=" * 60)

    # Create temp directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="batch_size_test_"))
        cleanup_temp = True

    print(f"Temp directory: {temp_dir}")

    # Create test config with only the max seq_len stage
    test_config_path = create_test_config(original_config, max_stage, temp_dir)
    print(f"Test config: {test_config_path}")

    try:
        best_bs, history = binary_search_batch_size(
            config_path=test_config_path,
            min_bs=args.min_batch_size,
            max_bs=args.max_batch_size,
            temp_dir=temp_dir,
            global_config=args.global_config,
            init_checkpoint=args.init_checkpoint,
            log_every=args.log_every,
            wait_for_logs=args.wait_for_logs,
            timeout=args.timeout,
            python_path=args.python,
            verbose=args.verbose,
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        print("\nTest History:")
        for bs, success, msg in sorted(history, key=lambda x: x[0]):
            if success:
                status = "OK"
            elif "OOM" in msg:
                status = "OOM"
            elif "Config error" in msg:
                status = "CFG"
            else:
                status = "FAIL"
            print(f"  batch_size={bs:>4}: [{status:>4}] {msg}")

        print("\n" + "-" * 60)
        if best_bs > 0:
            print(f"OPTIMAL BATCH SIZE: {best_bs}")
            print(f"  (tested at seq_len={max_seq_len})")
            print("-" * 60)

            # Suggest power-of-2 sizes
            power_of_2 = 1
            while power_of_2 * 2 <= best_bs:
                power_of_2 *= 2
            if power_of_2 != best_bs:
                print(f"Nearest power-of-2 batch size: {power_of_2}")

            # Suggest effective batch size with gradient accumulation
            print("\nSuggested configurations:")
            for target_effective in [32, 64, 128, 256]:
                if target_effective >= best_bs:
                    accum = (target_effective + best_bs - 1) // best_bs
                    actual_effective = best_bs * accum
                    print(
                        f"  batch_size={best_bs}, gradient_accumulation={accum} "
                        f"-> effective_batch_size={actual_effective}"
                    )
        else:
            print("ERROR: Could not find a working batch size.")
            print("-" * 60)

    finally:
        if cleanup_temp and temp_dir.exists():
            print(f"\nCleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
