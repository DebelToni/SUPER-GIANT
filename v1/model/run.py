"""
tmux_chaos_spawner.py

Chaos spawner for Train_model.py with:
 - spawn a pane every `--split-interval` seconds,
 - kill the *smallest* pane every `--kill-interval` seconds,
 - swap two random panes every `--swap-interval` seconds,
 - keep spawned panes open after the command finishes.

Run this script *inside* a tmux session/window.

Example:
  python3 tmux_chaos_spawner.py
  python3 tmux_chaos_spawner.py --split-interval 3 --kill-interval 6 --swap-interval 9
"""

import argparse
import os
import random
import shlex
import subprocess
import sys
import time
from typing import List, Dict, Optional, Tuple

time.sleep(5)

TMUX_BIN = "tmux"
DEFAULT_SPLIT_INTERVAL = 1.5
DEFAULT_KILL_INTERVAL = 2
DEFAULT_SWAP_INTERVAL = 1



# ---- tmux helper wrappers ----

def run_tmux(args: List[str], capture: bool = True) -> str:
    cmd = [TMUX_BIN] + args
    if capture:
        out = subprocess.check_output(cmd, text=True)
        return out.strip()
    else:
        subprocess.check_call(cmd)
        return ""


def ensure_inside_tmux() -> None:
    if "TMUX" not in os.environ:
        sys.exit("ERROR: Not inside tmux. Start a tmux session first (e.g., `tmux`), then run this script.")
    try:
        run_tmux(["display-message", "-p", "#{session_name}"])
    except subprocess.CalledProcessError:
        sys.exit("ERROR: tmux seems unavailable or this is not a tmux session.")


def current_window_id() -> str:
    return run_tmux(["display-message", "-p", "#{window_id}"])


def current_pane_id() -> str:
    # e.g. %1
    return os.environ.get("TMUX_PANE", "").strip()


def list_panes(window_id: str) -> List[str]:
    out = run_tmux(["list-panes", "-t", window_id, "-F", "#{pane_id}"])
    return [p.strip() for p in out.splitlines() if p.strip()]


def list_panes_info(window_id: str) -> List[Dict]:
    """
    Return a list of dicts with keys:
      - pane_id (e.g. %3)
      - width (int)
      - height (int)
      - area (width * height)
      - active (1/0)
    """
    fmt = "#{pane_id} #{pane_width} #{pane_height} #{pane_active}"
    out = run_tmux(["list-panes", "-t", window_id, "-F", fmt])
    entries = []
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        pane_id, w, h, active = parts[0], parts[1], parts[2], parts[3]
        try:
            wi = int(w); hi = int(h); ai = int(active)
            entries.append({
                "pane_id": pane_id,
                "width": wi,
                "height": hi,
                "area": wi * hi,
                "active": ai,
            })
        except ValueError:
            continue
    return entries


# ---- payload builder ----

def build_persistent_shell_payload(user_cmd: str) -> str:
    """
    Run user_cmd, echo exit code, then keep an interactive shell open in the pane.
    This prevents panes from closing immediately when the command exits with error.
    """
    safe_cmd = user_cmd
    payload = (
        "set -o pipefail; "
        f"{safe_cmd}; "
        "code=$?; "
        # "echo; echo \"[chaos] command exit code: $code — keeping pane open.\"; "
        "exec \"${SHELL:-bash}\" -i"
    )
    return payload


# ---- core actions ----

def split_random_direction(target_pane: str, command: str) -> Optional[str]:
    direction = random.choice(["left", "right", "up", "down"])
    flags = []
    if direction in ("left", "right"):
        flags.append("-h")
    else:
        flags.append("-v")
    if direction in ("left", "up"):
        flags.append("-b")

    persistent_payload = build_persistent_shell_payload(command)

    try:
        new_pane_id = run_tmux([
            "split-window",
            "-t", target_pane,
            "-P",
            "-F", "#{pane_id}",
            *flags,
            "--",
            "bash", "-lc", persistent_payload
        ])
        print(f"[spawn] split {direction:>5} from {target_pane} -> new {new_pane_id}")
        return new_pane_id
    except subprocess.CalledProcessError as e:
        print(f"[spawn] failed to split from {target_pane}: {e}")
        return None


def kill_smallest_pane(window_id: str, protect: Optional[str]) -> None:
    panes_info = list_panes_info(window_id)
    if protect:
        panes_info = [p for p in panes_info if p["pane_id"] != protect]
    if len(panes_info) <= 1:
        return
    # find smallest area
    min_area = min(p["area"] for p in panes_info)
    candidates = [p for p in panes_info if p["area"] == min_area]
    victim = random.choice(candidates)
    victim_id = victim["pane_id"]
    try:
        run_tmux(["kill-pane", "-t", victim_id], capture=False)
        print(f"[kill ] removed smallest pane {victim_id} (area={victim['area']})")
    except subprocess.CalledProcessError as e:
        print(f"[kill ] failed to kill {victim_id}: {e}")


def swap_two_random_panes(window_id: str, protect: Optional[str]) -> None:
    panes_info = list_panes_info(window_id)
    # exclude protected pane if given
    if protect:
        panes_info = [p for p in panes_info if p["pane_id"] != protect]
    pane_ids = [p["pane_id"] for p in panes_info]
    if len(pane_ids) < 2:
        return
    a, b = random.sample(pane_ids, 2)
    try:
        run_tmux(["swap-pane", "-s", a, "-t", b], capture=False)
        print(f"[swap ] swapped panes {a} <-> {b}")
    except subprocess.CalledProcessError as e:
        print(f"[swap ] failed to swap {a} and {b}: {e}")


# ---- defaults ----

def default_cmd() -> str:
    py = shlex.quote(sys.executable)
    script = shlex.quote("./Train_model.py")
    return f"{py} {script}"


# ---- main loop ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default=None,
                        help="Shell command for each new pane (default: 'python ./Train_model.py').")
    parser.add_argument("--allow-kill-self", action="store_true",
                        help="Allow killing the pane that runs this spawner.")
    parser.add_argument("--split-interval", type=float, default=DEFAULT_SPLIT_INTERVAL,
                        help="Seconds between spawns (default: 3).")
    parser.add_argument("--kill-interval", type=float, default=DEFAULT_KILL_INTERVAL,
                        help="Seconds between random kills (default: 6).")
    parser.add_argument("--swap-interval", type=float, default=DEFAULT_SWAP_INTERVAL,
                        help="Seconds between swapping two random panes (default: 9).")
    args = parser.parse_args()

    ensure_inside_tmux()
    win = current_window_id()

    # keep panes visible on exit
    try:
        run_tmux(["set-option", "-w", "remain-on-exit", "on"], capture=False)
    except subprocess.CalledProcessError:
        pass

    self_pane = current_pane_id() if not args.allow_kill_self else None
    cmd = args.cmd or default_cmd()

    print(f"[info ] window={win} self_pane={self_pane or '(killable)'}")
    print(f"[info ] spawn every {args.split_interval}s; kill every {args.kill_interval}s; swap every {args.swap_interval}s")
    print(f"[info ] command for new panes: {cmd}")

    last_spawn = 0.0
    last_kill = 0.0
    last_swap = 0.0

    try:
        while True:
            now = time.monotonic()

            if now - last_spawn >= args.split_interval:
                panes = list_panes(win)
                target = random.choice(panes) if panes else (self_pane or "%0")
                split_random_direction(target, cmd)
                last_spawn = now

            if now - last_kill >= args.kill_interval:
                kill_smallest_pane(win, protect=self_pane)
                last_kill = now

            if now - last_swap >= args.swap_interval:
                swap_two_random_panes(win, protect=self_pane)
                last_swap = now

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[exit ] Ctrl-C received, stopping spawner.")


if __name__ == "__main__":
    main()

