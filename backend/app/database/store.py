"""Persistence for searches, route comparisons, experiments and training runs.

* ``MongoStore``  - MongoDB Atlas (``MONGODB_URI``), the production store.
* ``LocalStore``  - JSON-lines files under data/runtime/, used automatically
  when no MongoDB URI is configured or Atlas is unreachable.  The health
  endpoint always reports which store is active.

Collections: searches, route_comparisons, experiments, training_runs.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from reinforcement_learning.environment.config import RUNTIME_DIR

log = logging.getLogger("store")
COLLECTIONS = ("searches", "route_comparisons", "experiments", "training_runs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LocalStore:
    kind = "local-file"

    def __init__(self, folder: Path = RUNTIME_DIR, reason: str = ""):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.reason = reason
        self._lock = threading.Lock()

    def _file(self, col: str) -> Path:
        return self.folder / f"{col}.jsonl"

    def insert(self, col: str, doc: dict) -> str:
        doc = {"_id": uuid.uuid4().hex, "created_at": _now(), **doc}
        with self._lock, open(self._file(col), "a") as fh:
            fh.write(json.dumps(doc, default=str) + "\n")
        return doc["_id"]

    def find(self, col: str, limit: int = 100, filt: dict | None = None) -> list[dict]:
        f = self._file(col)
        if not f.exists():
            return []
        out = []
        with self._lock, open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if filt and any(d.get(k) != v for k, v in filt.items()):
                    continue
                out.append(d)
        return list(reversed(out))[:limit]

    def count(self, col: str) -> int:
        f = self._file(col)
        if not f.exists():
            return 0
        with open(f) as fh:
            return sum(1 for _ in fh)

    def status(self) -> dict:
        return {"type": self.kind, "connected": True, "location": str(self.folder),
                "note": self.reason or "MONGODB_URI not set - using local JSON-lines files (not persistent on Render)"}


class MongoStore:
    kind = "mongodb"

    def __init__(self, uri: str, db: str):
        from pymongo import DESCENDING, MongoClient

        self._desc = DESCENDING
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="smartroute-ai")
        self.client.admin.command("ping")
        self.db = self.client[db]
        for c in COLLECTIONS:
            self.db[c].create_index([("created_at", DESCENDING)])

    def insert(self, col: str, doc: dict) -> str:
        doc = {"created_at": _now(), **doc}
        res = self.db[col].insert_one(json.loads(json.dumps(doc, default=str)))
        return str(res.inserted_id)

    def find(self, col: str, limit: int = 100, filt: dict | None = None) -> list[dict]:
        cur = self.db[col].find(filt or {}).sort("created_at", self._desc).limit(limit)
        out = []
        for d in cur:
            d["_id"] = str(d["_id"])
            out.append(d)
        return out

    def count(self, col: str) -> int:
        return self.db[col].estimated_document_count()

    def status(self) -> dict:
        t = time.perf_counter()
        try:
            self.client.admin.command("ping")
            ok = True
        except Exception:
            ok = False
        return {"type": self.kind, "connected": ok, "database": self.db.name,
                "ping_ms": round((time.perf_counter() - t) * 1000, 1)}


_store = None
_store_lock = threading.Lock()


def get_store():
    global _store
    with _store_lock:
        if _store is None:
            from backend.app.core.settings import settings

            if settings.mongodb_uri:
                try:
                    _store = MongoStore(settings.mongodb_uri, settings.mongodb_db)
                    log.info("Connected to MongoDB database '%s'", settings.mongodb_db)
                except Exception as exc:
                    log.error("MongoDB unavailable (%s) - falling back to local files", exc)
                    _store = LocalStore(reason=f"MongoDB connection failed: {exc.__class__.__name__}")
            else:
                _store = LocalStore()
        return _store


def reset_store_for_tests(store) -> None:
    global _store
    _store = store
