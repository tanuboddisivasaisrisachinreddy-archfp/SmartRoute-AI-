"""Background training jobs.

Training runs in a separate OS process (the same CLI scripts used offline),
so it never blocks API requests and its memory is released when it ends.
Progress is exchanged through a JSON status file.  When a job completes the
engine hot-reloads the new checkpoints.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

from reinforcement_learning.environment.config import CHECKPOINT_DIR, REPO_ROOT, RUNTIME_DIR

log = logging.getLogger("training")


class TrainingJobManager:
    def __init__(self, on_complete=None, store=None):
        self.proc: subprocess.Popen | None = None
        self.job: dict | None = None
        self.on_complete = on_complete
        self.store = store
        self.lock = threading.Lock()
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, algorithm: str, params: dict) -> dict:
        with self.lock:
            if self.running():
                raise RuntimeError("A training job is already running")
            progress = RUNTIME_DIR / f"training_{algorithm}.json"
            progress.unlink(missing_ok=True)
            if algorithm == "q_learning":
                cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train_q_learning.py"),
                       "--episodes", str(params["episodes"]), "--alpha", str(params["learning_rate"]),
                       "--gamma", str(params["discount_factor"]), "--epsilon", str(params["epsilon"]),
                       "--seed", str(params["seed"]), "--preferences", ",".join(params["preferences"]),
                       "--goals", params.get("goals") or "all",
                       "--output", str(CHECKPOINT_DIR / "q_learning.npz")]
            elif algorithm == "dqn":
                cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train_dqn.py"),
                       "--episodes", str(params["episodes"]), "--lr", str(params["learning_rate"]),
                       "--gamma", str(params["discount_factor"]), "--epsilon", str(params["epsilon"]),
                       "--batch-size", str(params["batch_size"]), "--buffer-size", str(params["buffer_size"]),
                       "--seed", str(params["seed"]), "--output", str(CHECKPOINT_DIR / "dqn")]
            else:
                raise ValueError("algorithm must be q_learning or dqn")
            cmd += ["--progress-file", str(progress)]
            logf = open(RUNTIME_DIR / f"training_{algorithm}.log", "w")
            self.proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT)
            self.job = {"id": f"{algorithm}-{int(time.time())}", "algorithm": algorithm, "params": params,
                        "progress_file": str(progress), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "pid": self.proc.pid, "command": " ".join(Path(c).name if i < 2 else c
                                                                  for i, c in enumerate(cmd))}
            threading.Thread(target=self._watch, daemon=True).start()
            log.info("Started training job %s", self.job["id"])
            return self.status()

    def stop(self) -> dict:
        with self.lock:
            if self.running():
                self.proc.terminate()
                try:
                    self.proc.wait(10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                if self.job:
                    self.job["stopped"] = True
        return self.status()

    def _watch(self) -> None:
        proc, job = self.proc, self.job
        code = proc.wait()
        job["exit_code"] = code
        job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        prog = self._read_progress(job)
        if code == 0 and self.on_complete:
            try:
                self.on_complete()
                job["models_reloaded"] = True
            except Exception as exc:
                job["reload_error"] = str(exc)
        if self.store is not None:
            try:
                self.store.insert("training_runs", {**{k: v for k, v in job.items() if k != "progress_file"},
                                                    "final": {k: v for k, v in (prog or {}).items() if k != "curve"},
                                                    "curve": (prog or {}).get("curve", [])})
            except Exception as exc:
                log.warning("could not store training run: %s", exc)

    @staticmethod
    def _read_progress(job: dict | None) -> dict | None:
        if not job:
            return None
        p = Path(job["progress_file"])
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def status(self) -> dict:
        prog = self._read_progress(self.job)
        if self.job is None:
            state = "idle"
        elif self.running():
            state = "running"
        elif self.job.get("stopped"):
            state = "stopped"
        elif self.job.get("exit_code") == 0:
            state = "completed"
        elif self.job.get("exit_code") is None:
            state = "starting"
        else:
            state = "failed"
        log_tail = None
        if self.job and state in ("failed", "stopped"):
            lf = RUNTIME_DIR / f"training_{self.job['algorithm']}.log"
            if lf.exists():
                log_tail = lf.read_text()[-1500:]
        return {"state": state, "job": self.job, "progress": prog, "log_tail": log_tail}
