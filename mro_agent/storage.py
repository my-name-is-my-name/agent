import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from contextlib import contextmanager


class Conflict(Exception):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS assessments (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, chat_id TEXT NOT NULL,
                    UNIQUE(user_id, chat_id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, assessment_id TEXT NOT NULL, message_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
                    result TEXT, created REAL NOT NULL, UNIQUE(assessment_id, message_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    stage TEXT NOT NULL, status TEXT NOT NULL, message TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS job_events ON events(job_id, seq);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def submit(self, user_id, chat_id, message_id, payload):
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM assessments WHERE user_id=? AND chat_id=?", (user_id, chat_id)).fetchone()
            aid = row["id"] if row else uuid.uuid4().hex
            if not row:
                db.execute("INSERT INTO assessments VALUES (?,?,?)", (aid, user_id, chat_id))
            existing = db.execute("SELECT * FROM jobs WHERE assessment_id=? AND message_id=?", (aid, message_id)).fetchone()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise Conflict("message_id already has different content")
                return dict(existing)
            if db.execute("SELECT 1 FROM jobs WHERE assessment_id=? AND status IN ('queued','running')", (aid,)).fetchone():
                raise Conflict("assessment is already running; wait for its result")
            job_id = uuid.uuid4().hex
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", (job_id, aid, message_id, fingerprint, encoded, "queued", None, time.time()))
            return dict(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def job(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT j.*, a.user_id FROM jobs j JOIN assessments a ON a.id=j.assessment_id WHERE j.id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def assessment(self, aid, user_id):
        with self.connect() as db:
            row = db.execute("SELECT id FROM assessments WHERE id=? AND user_id=?", (aid, user_id)).fetchone()
            if not row:
                return None
            jobs = [dict(r) for r in db.execute("SELECT id, status, result, created FROM jobs WHERE assessment_id=? ORDER BY created", (aid,))]
        return {"id": aid, "jobs": [{**j, "result": json.loads(j["result"]) if j["result"] else None} for j in jobs]}

    def latest_result(self, aid, exclude):
        with self.connect() as db:
            row = db.execute("SELECT result FROM jobs WHERE assessment_id=? AND id<>? AND result IS NOT NULL AND status='completed' ORDER BY created DESC LIMIT 1", (aid, exclude)).fetchone()
        return json.loads(row["result"]) if row else None

    def claim_next(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
                return row["id"]
        return None

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='queued' WHERE status='running'")

    def finish(self, job_id, result, failed=False):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,result=? WHERE id=?", ("failed" if failed else "completed", json.dumps(result, ensure_ascii=False), job_id))

    def event(self, job_id, stage, status, message):
        with self.connect() as db:
            db.execute("INSERT INTO events(job_id,stage,status,message,created) VALUES (?,?,?,?,?)", (job_id, stage, status, message, time.time()))

    def events(self, job_id, after=0):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM events WHERE job_id=? AND seq>? ORDER BY seq", (job_id, after))]
