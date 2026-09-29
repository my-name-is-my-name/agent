import asyncio
import json
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from .config import Settings
from .models import AssessmentRequest, ChatRequest
from .service import Service
from .storage import Conflict


def create_app(settings=None, service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.settings = settings or Settings.from_env()
        app.state.service = service or Service(app.state.settings)
        yield
        if service is None:
            app.state.service.close()

    app = FastAPI(title="MRO Assessment Agent", version="0.1.0", lifespan=lifespan)

    def auth(authorization: str = Header(default=""), x_user_id: str = Header(default="")):
        if not secrets.compare_digest(authorization.encode("utf-8"), ("Bearer " + app.state.settings.api_token).encode("utf-8")):
            raise HTTPException(401, "Invalid service token")
        if x_user_id not in app.state.settings.allowed_users:
            raise HTTPException(403, "User is outside the authorized pilot group")
        return x_user_id

    def owned_job(job_id, user):
        job = app.state.service.store.job(job_id)
        if not job or job["user_id"] != user:
            raise HTTPException(404, "Job not found")
        return job

    def submit(user, request):
        try:
            return app.state.service.submit(user, request)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/health")
    def health():
        # No upstream health requests: some existing health implementations write state.
        worker = app.state.service.worker
        return {"status": "ok" if worker.is_alive() else "degraded", "version": "0.1.0", "worker_alive": worker.is_alive(), "dependencies": "not_probed"}

    @app.get("/v1/models")
    def models(user=Depends(auth)):
        return {"object": "list", "data": [{"id": "mro-assessment-agent", "object": "model", "owned_by": "local"}]}

    @app.post("/api/assessments", status_code=202)
    def create(request: AssessmentRequest, user=Depends(auth)):
        job = submit(user, request)
        return {"assessment_id": job["assessment_id"], "job_id": job["id"], "status": job["status"]}

    @app.get("/api/assessments/{assessment_id}")
    def assessment(assessment_id: str, user=Depends(auth)):
        result = app.state.service.store.assessment(assessment_id, user)
        if not result:
            raise HTTPException(404, "Assessment not found")
        return result

    @app.post("/api/assessments/{assessment_id}/answers", status_code=202)
    def answers(assessment_id: str, request: AssessmentRequest, user=Depends(auth)):
        if not app.state.service.store.assessment(assessment_id, user):
            raise HTTPException(404, "Assessment not found")
        # Validate chat binding before queuing an answer.
        with app.state.service.store.connect() as db:
            row = db.execute("SELECT chat_id FROM assessments WHERE id=?", (assessment_id,)).fetchone()
        if row["chat_id"] != request.chat_id:
            raise HTTPException(409, "Chat does not belong to this assessment")
        job = submit(user, request)
        return {"assessment_id": job["assessment_id"], "job_id": job["id"], "status": job["status"]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, user=Depends(auth)):
        job = owned_job(job_id, user)
        return {"id": job_id, "status": job["status"], "result": json.loads(job["result"]) if job["result"] else None}

    @app.get("/api/jobs/{job_id}/events")
    def events(job_id: str, after: int = 0, user=Depends(auth)):
        owned_job(job_id, user)
        return {"events": app.state.service.store.events(job_id, after)}

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest, user=Depends(auth)):
        latest = next((m for m in reversed(request.messages) if m.role == "user"), None)
        if latest is None:
            raise HTTPException(400, "User text required")
        attachment = request.attachments_present
        if isinstance(latest.content, str):
            text = latest.content.strip()
        elif isinstance(latest.content, list):
            text = "\n".join(str(p.get("text", "")) for p in latest.content if isinstance(p, dict) and p.get("type") == "text").strip()
            attachment |= any(not isinstance(p, dict) or p.get("type") != "text" for p in latest.content)
        else:
            raise HTTPException(400, "Unsupported user content")
        if not text or len(text) > 16000:
            raise HTTPException(400, "User text must contain 1–16000 characters")
        job = submit(user, AssessmentRequest(request=text, chat_id=request.chat_id, message_id=request.message_id, attachments_present=attachment))

        def envelope(delta, finish=None):
            return {"id": "chatcmpl-" + job["id"], "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        async def stream():
            def data(obj):
                return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
            yield data(envelope({"role": "assistant"}))
            seq, heartbeat = 0, time.monotonic()
            while True:
                for event in app.state.service.store.events(job["id"], seq):
                    seq = event["seq"]
                    yield data(envelope({"reasoning_content": event["message"] + "\n"}))
                current = app.state.service.store.job(job["id"])
                if current["status"] in ("completed", "failed"):
                    # Drain final events committed between the previous event read and job read.
                    for event in app.state.service.store.events(job["id"], seq):
                        seq = event["seq"]
                        yield data(envelope({"reasoning_content": event["message"] + "\n"}))
                    result = json.loads(current["result"])
                    yield data(envelope({"content": result["content"] + "\n\nКарточка: " + job["assessment_id"]}))
                    yield data(envelope({}, "stop"))
                    yield "data: [DONE]\n\n"
                    break
                if time.monotonic() - heartbeat >= 10:
                    yield ": keepalive\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(0.1)

        if request.stream:
            return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        while True:
            current = app.state.service.store.job(job["id"])
            if current["status"] in ("completed", "failed"):
                result = json.loads(current["result"])
                return {"id": "chatcmpl-" + job["id"], "object": "chat.completion", "created": int(time.time()), "model": request.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]}, "finish_reason": "stop"}], "assessment_id": job["assessment_id"]}
            await asyncio.sleep(0.1)

    return app
