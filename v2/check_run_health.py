#!/usr/bin/env python3
"""
Liveness check for a long-running background job — process alive + log advancing.

Written after a real incident (2026-08-17): a 300-step GRPO training run was
killed externally partway through (most likely a session/context teardown, not a
training bug), and the log file was mistaken for live progress for two status
updates in a row because `ps aux | grep <script>` was misread and GPU memory
usage (which turned out to belong to an unrelated process) was used as a stand-in
for liveness instead of directly confirming the PID. This script exists so that
check is a five-second command instead of manual, error-prone inference next time.

Neither signal alone is enough:
  - Process alive but log stale  -> genuinely stuck/hung (or between infrequent
    log lines — check the expected logging interval before assuming the worst).
  - Process dead but log "recent" -> possible if the log was touched by something
    else; rare, but why both checks matter rather than either alone.
  - Process dead AND log stale    -> the run has stopped. Check the log tail for
    a traceback; if there isn't one, it was likely killed externally, not crashed.

Usage:
  python3 check_run_health.py --pid 22501 --log grpo_real_run_300steps_resumed.log
  python3 check_run_health.py --pid 22501 --log foo.log --stale-minutes 5

Exit code 0 if healthy (alive + log recent), 1 otherwise — usable in a loop:
  while python3 check_run_health.py --pid $PID --log run.log; do sleep 120; done
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: no-op, just checks the PID exists and we can signal it
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else — still alive


def process_state(pid: int) -> str | None:
    """Best-effort process state letter (R/S/D/Z/T/...) via `ps`, None if not found."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat=", "--no-headers"],
            capture_output=True, text=True, timeout=5,
        )
        state = out.stdout.strip()
        return state or None
    except Exception:
        return None


def log_staleness_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    return time.time() - log_path.stat().st_mtime


def log_has_completion_marker(log_path: Path, marker: str, tail_bytes: int = 20000) -> bool:
    """Check the tail of the log for a marker indicating a clean finish (e.g.
    "train_runtime", which HF Trainer prints as the final training-summary dict
    key). Without this, a process that finished normally and a process that was
    killed mid-run both look identical to a bare PID+mtime check: "process gone,
    log stale.\""""
    if not log_path.exists():
        return False
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            tail = f.read().decode("utf-8", errors="replace")
        return marker in tail
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", type=int, required=True, help="PID of the job to check")
    parser.add_argument("--log", type=Path, required=True, help="Path to the job's log file")
    parser.add_argument("--stale-minutes", type=float, default=10.0,
                         help="Log mtime older than this (minutes) is flagged as stale (default: 10)")
    parser.add_argument("--completion-marker", default="train_runtime",
                         help="String to look for in the log tail indicating a clean "
                              "finish (default: 'train_runtime', HF Trainer's final "
                              "summary-dict key — this project's training scripts all "
                              "print it). Set to '' to disable this check.")
    args = parser.parse_args()

    alive = is_process_alive(args.pid)
    state = process_state(args.pid) if alive else None
    staleness = log_staleness_seconds(args.log)

    print(f"process (pid={args.pid}): {'ALIVE' if alive else 'NOT FOUND'}"
          + (f" (state={state})" if state else ""))

    if staleness is None:
        print(f"log ({args.log}): NOT FOUND")
        log_ok = False
    else:
        stale_threshold_s = args.stale_minutes * 60
        log_ok = staleness < stale_threshold_s
        mins = staleness / 60
        print(f"log ({args.log}): last write {mins:.1f} min ago "
              f"({'OK' if log_ok else f'STALE — older than {args.stale_minutes:.0f} min threshold'})")

    completed = False
    if args.completion_marker and args.log.exists():
        completed = log_has_completion_marker(args.log, args.completion_marker)
        print(f"completion marker ('{args.completion_marker}'): "
              f"{'FOUND — looks like a clean finish' if completed else 'not found'}")

    healthy = alive and log_ok
    print()
    if not alive and completed:
        print("FINISHED — process is gone but the log shows a clean completion "
              "marker. This is the expected end state for a run that finished "
              "normally, not a failure.")
        sys.exit(0)
    elif healthy:
        print("HEALTHY — process alive and log advancing.")
    elif alive and not log_ok:
        print("WARNING — process is alive but the log hasn't advanced. Possibly "
              "stuck/hung, or between infrequent log lines (check expected "
              "logging interval before assuming the worst).")
    elif not alive and log_ok:
        print("WARNING — log looks recent but the process isn't found. Something "
              "else may have touched the log, or the process just exited — check "
              "the log tail for a clean completion message vs. mid-output cutoff.")
    else:
        print("DEAD — process not found, log is stale, and no completion marker "
              "found. Check the log tail for a traceback; if there isn't one, it "
              "was likely killed externally (session/context teardown, "
              "OOM-killer, manual kill), not crashed.")

    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
