import json
import socket
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mro_agent.api import create_app
from mro_agent.clients import JsonHTTP, ToolError, Tools, validate_internal_url
from mro_agent.config import Settings
from mro_agent.service import Service
from mro_agent.storage import Conflict
from mro_agent.models import AssessmentRequest
from mro_agent.workflow import qualified_candidates


TEXT = "Выполнен анализ повреждения. Выпущен отчёт об анализе."


@pytest.mark.parametrize("field", ["content", "reasoning_content"])
def test_llm_sends_json_schema(settings, monkeypatch, field):
    from mro_agent.models import Profile

    tools = Tools(settings)
    def respond(method, url, payload, token):
        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "Profile", "schema": Profile.model_json_schema()},
        }
        return {"choices": [{"finish_reason": "stop", "message": {field: "{}"}}]}

    monkeypatch.setattr(tools.http, "call", respond)
    try:
        assert tools.llm("test", {"request": "test"}, Profile) == Profile().model_dump()
    finally:
        tools.close()


class FakeTools:
    """Synthetic documents only. No network or production corpus."""
    def __init__(self):
        self.calls = []
        self.empty = False
        self.unavailable = False
        self.no_documents = False
        self.unresolved = False
        self.bad_claim = False
        self.reject = False
        self.long = False
        self.mismatch = False
        self.closed = False

    def llm(self, prompt, data, schema):
        name = schema.__name__
        self.calls.append(name)
        if name == "Profile":
            enough = "панели" in data["request"]
            return {"object": "панель" if enough else "", "task": "анализ повреждения", "expected_result": "отчёт", "identifiers": [], "questions": [] if enough else ["Укажите объект работ."]}
        if name == "Extraction":
            source = data.get("source") or data.get("sources", [{}])[0]
            return {"claims": [{"text": "Выполнен анализ повреждения", "quote": "Выдуманная цитата" if self.bad_claim else "Выполнен анализ повреждения.", "evidence_id": source.get("id", "E1"), "category": "work"}]}
        if name == "Verdict":
            return {"supported": not self.reject}
        if name == "Synthesis":
            return {"proposed_scope": [{"text": "Рассмотреть анализ повреждения", "evidence_ids": ["E1"]}], "questions": ["Уточните применимость исходных данных."]}
        if name == "Followup":
            message = data["message"].casefold()
            if message == "коррозия rib5":
                return {"kind": "ambiguous"}
            return {"kind": "question" if "?" in message else "correction"}
        raise AssertionError(name)

    def search(self, text, profile, user_id):
        self.calls.append("search")
        if self.unavailable:
            raise ToolError("temporarily_unavailable")
        if self.empty:
            return {"status": "ok", "similarity_status": "no_qualified_matches", "accepted": [], "not_accepted": []}
        return {"status": "ok", "similarity_status": "qualified_matches_found", "accepted": [{"case_id": "DEMO-1", "qualified": True, "score": 0.9, "similarity_reason_class": "same_work_type", "reasons": ["Совпадает вид работы"]}], "not_accepted": []}

    def resolve(self, case_id, user_id):
        self.calls.append("resolve")
        if self.unresolved:
            raise ToolError("case_mapping_unresolved")
        return "INTERNAL-1"

    def case(self, case_id, user_id):
        return {"case_id": case_id, "documents": [] if self.no_documents else [{"document_id": "DOC-1", "title": "Отчёт"}]}

    def document(self, document_id, user_id):
        self.calls.append("document")
        return {"document_id": document_id, "case_id": "WRONG" if self.mismatch else "INTERNAL-1", "title": "Синтетический отчёт", "revision": "A", "chunks": [{"chunk_id": "CHUNK-1", "text": TEXT * (300 if self.long else 1), "page_number": 1}]}

    def close(self):
        self.closed = True


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "runtime", "x" * 32, frozenset({"alice", "bob"}), "http://localhost:8121", "http://localhost:1234/v1", "synthetic-model")


@pytest.fixture
def service(settings):
    result = Service(settings, FakeTools(), start_worker=False)
    yield result
    result.close()


def run(service, text="Анализ повреждения панели", message="m1", user="alice", chat="chat1", attachments=False):
    job = service.submit(user, AssessmentRequest(request=text, chat_id=chat, message_id=message, attachments_present=attachments))
    if job["status"] == "queued":
        claimed = service.store.claim_next()
        assert claimed == job["id"]
        service.process(claimed)
    return service.store.job(job["id"])


def result(job):
    return json.loads(job["result"])


def test_full_workflow(service):
    job = run(service)
    assert job["status"] == "completed", job["result"]
    value = result(job)
    assert value["status"] == "ready"
    assert value["state"]["claims"][0]["quote"] in TEXT
    assert "DOC-1" in value["content"]
    assert "гипотезы" in value["content"]
    assert "не рассчитывает часы" in value["content"]
    stages = [e["stage"] for e in service.store.events(job["id"])]
    assert stages.index("search") < stages.index("read") < stages.index("extract") < stages.index("finish")


def test_clarification_survives_restart(settings):
    first = Service(settings, FakeTools(), start_worker=False)
    job = run(first, "Нужен анализ")
    assert result(job)["status"] == "waiting"
    assert "search" not in first.tools.calls
    aid = job["assessment_id"]
    first.close()
    second = Service(settings, FakeTools(), start_worker=False)
    try:
        resumed = run(second, "Повреждение панели", "m2")
        assert resumed["assessment_id"] == aid
        assert result(resumed)["status"] == "ready", resumed["result"]
        assert second.tools.calls.count("search") == 1
    finally:
        second.close()


def test_optional_questions_do_not_block_search(service, monkeypatch):
    original = service.tools.llm
    def llm(prompt, data, schema):
        value = original(prompt, data, schema)
        if schema.__name__ == "Profile":
            value["questions"] = ["Каковы размеры повреждения?", "Назовите номера исторических заявок."]
        return value
    monkeypatch.setattr(service.tools, "llm", llm)
    value = result(run(service))
    assert value["status"] == "ready"
    assert service.tools.calls.count("search") == 1
    assert value["state"]["questions"] == []
    assert "не блокирует поиск" in value["content"]


@pytest.mark.parametrize("missing", ["object", "task"])
def test_missing_prerequisite_blocks_even_without_model_questions(service, monkeypatch, missing):
    original = service.tools.llm
    def llm(prompt, data, schema):
        value = original(prompt, data, schema)
        if schema.__name__ == "Profile":
            value[missing] = "   "
            value["questions"] = []
        return value
    monkeypatch.setattr(service.tools, "llm", llm)
    assert result(run(service))["status"] == "waiting"
    assert "search" not in service.tools.calls


def test_last_clarification_resumes_search_after_restart(settings, monkeypatch):
    first = Service(settings, FakeTools(), start_worker=False)
    try:
        for index in range(3):
            job = run(first, "Нужна оценка", "m" + str(index))
            assert result(job)["status"] == "waiting"
        aid = job["assessment_id"]
    finally:
        first.close()
    second = Service(settings, FakeTools(), start_worker=False)
    original = second.tools.llm
    def llm(prompt, data, schema):
        value = original(prompt, data, schema)
        if schema.__name__ == "Profile":
            assert "Нужна оценка" in data["request"]
            value["questions"] = ["Уточните размеры повреждения."]
        return value
    monkeypatch.setattr(second.tools, "llm", llm)
    try:
        job = run(second, "Анализ повреждения панели; размеры неизвестны", "m3")
        assert job["assessment_id"] == aid
        assert result(job)["status"] == "ready"
        assert second.tools.calls.count("search") == 1
    finally:
        second.close()


@pytest.mark.parametrize("flag,expected", [("empty", "no_qualified_matches"), ("unavailable", "temporarily_unavailable"), ("no_documents", "отсутствуют"), ("unresolved", "case_mapping_unresolved"), ("mismatch", "document_case_mismatch")])
def test_missing_evidence_is_not_success_or_decline(service, flag, expected):
    setattr(service.tools, flag, True)
    value = result(run(service))
    assert not value["state"]["claims"]
    assert expected in " ".join(value["state"]["warnings"])
    assert "Достаточных подтверждений не получено" in value["content"]


@pytest.mark.parametrize("flag", ["bad_claim", "reject"])
def test_unsupported_claim_removed(service, flag):
    setattr(service.tools, flag, True)
    value = result(run(service))
    assert value["state"]["claims"] == []
    assert value["state"]["proposal"]["proposed_scope"] == []


def test_partial_read_is_explicit(service):
    service.tools.long = True
    value = result(run(service, attachments=True))
    assert "только часть" in value["content"]
    assert "Вложения" in value["content"]
    assert sum(len(e["text"]) for e in value["state"]["evidence"]) <= service.settings.max_doc_chars


def test_idempotency_and_content_conflict(service):
    first = run(service)
    calls = len(service.tools.calls)
    repeated = run(service)
    assert first["id"] == repeated["id"]
    assert len(service.tools.calls) == calls
    with pytest.raises(Conflict):
        run(service, text="Другая заявка")


def test_users_and_chats_isolated(service):
    a = run(service)
    b = run(service, user="bob")
    c = run(service, chat="chat2")
    assert len({x["assessment_id"] for x in (a, b, c)}) == 3
    assert service.store.assessment(a["assessment_id"], "bob") is None


def test_followup_does_not_overwrite_request(service):
    first = result(run(service))
    question = result(run(service, "Какие работы выполнены?", "m2"))
    assert question["state"]["request"] == first["state"]["request"]
    assert service.tools.calls.count("search") == 1
    correction = result(run(service, "Добавлено повреждение второй панели", "m3"))
    assert "второй панели" in correction["state"]["request"]
    assert service.tools.calls.count("search") == 2


def test_explicit_new_request_is_not_merged_or_searched(service):
    first = result(run(service))
    followup = result(run(service, "это другая заявка: трещина frame 42", "m2"))
    assert followup["followup_kind"] == "new_request"
    assert followup["state"]["request"] == first["state"]["request"]
    assert "frame 42" not in followup["state"]["request"]
    assert service.tools.calls.count("search") == 1
    assert "отдельную заявку" in followup["content"]


def test_explicit_correction_is_merged_and_researched(service):
    run(service)
    followup = result(run(service, "добавь: повреждение также затрагивает stringer 12", "m2"))
    assert followup["followup_kind"] == "correction"
    assert "stringer 12" in followup["state"]["request"]
    assert service.tools.calls.count("search") == 2


def test_candidate_question_uses_saved_state_when_documents_unresolved(service, monkeypatch):
    def search(text, profile, user_id):
        service.tools.calls.append("search")
        return {
            "status": "ok", "similarity_status": "qualified_matches_found",
            "accepted": [
                {"case_id": case_id, "qualified": True, "score": score,
                 "similarity_reason_class": "same_work_type", "reasons": ["Совпадает вид работы"]}
                for case_id, score in (("MP-0776", 0.9), ("MP-0632.1", 0.8), ("MP-1061", 0.7))
            ],
            "not_accepted": [],
        }
    monkeypatch.setattr(service.tools, "search", search)
    service.tools.unresolved = True
    run(service)
    followup = result(run(service, "а какие аналоги ты нашел?", "m2"))
    assert followup["followup_kind"] == "question"
    assert service.tools.calls.count("search") == 1
    assert all(case_id in followup["content"] for case_id in ("MP-0776", "MP-0632.1", "MP-1061"))
    assert "case_mapping_unresolved" in followup["content"]
    assert "техническая применимость не подтверждена" in followup["content"]
    assert "Недостаточно данных в изученных источниках" not in followup["content"]


def test_ambiguous_engineering_message_does_not_modify_request(service):
    first = result(run(service))
    followup = result(run(service, "коррозия rib5", "m2"))
    assert followup["followup_kind"] == "ambiguous"
    assert followup["state"]["request"] == first["state"]["request"]
    assert service.tools.calls.count("search") == 1
    assert "или это отдельная заявка?" in followup["content"]


def test_document_question_uses_saved_state_without_research(service):
    run(service)
    followup = result(run(service, "Какие документы были изучены?", "m2"))
    assert followup["followup_kind"] == "question"
    assert service.tools.calls.count("search") == 1
    assert "DOC-1" in followup["content"]


def test_followup_routing_survives_service_restart(settings):
    first_service = Service(settings, FakeTools(), start_worker=False)
    try:
        first = result(run(first_service))
    finally:
        first_service.close()
    second_service = Service(settings, FakeTools(), start_worker=False)
    try:
        followup = result(run(second_service, "это другая заявка: трещина frame 42", "m2"))
        assert followup["followup_kind"] == "new_request"
        assert followup["state"]["request"] == first["state"]["request"]
        assert "search" not in second_service.tools.calls
    finally:
        second_service.close()


def test_weak_candidates_never_fill_output():
    payload = {"status": "ok", "similarity_status": "qualified_matches_found", "accepted": [{"case_id": "W", "qualified": True, "similarity_reason_class": "weak_analog"}, {"case_id": "N", "qualified": False}], "not_accepted": []}
    assert qualified_candidates(payload) == []


def test_queue_recovery_and_same_process_exclusion(settings):
    first = Service(settings, FakeTools(), start_worker=False)
    job = first.submit("alice", AssessmentRequest(request="Анализ повреждения панели", chat_id="c", message_id="m"))
    assert first.store.claim_next() == job["id"]
    with pytest.raises(RuntimeError):
        Service(settings, FakeTools(), start_worker=False)
    first.close()
    second = Service(settings, FakeTools(), start_worker=False)
    try:
        assert second.store.claim_next() == job["id"]
        second.process(job["id"])
        assert result(second.store.job(job["id"]))["status"] == "ready"
    finally:
        second.close()


def test_completed_checkpoint_not_reexecuted(service):
    job = run(service)
    count = len(service.tools.calls)
    # Simulate death between graph checkpoint and queue result commit.
    with service.store.connect() as db:
        db.execute("UPDATE jobs SET status='running',result=NULL WHERE id=?", (job["id"],))
    service.process(job["id"])
    assert len(service.tools.calls) == count
    assert result(service.store.job(job["id"]))["status"] == "ready"


def test_api_sse_auth_and_replay(settings):
    svc = Service(settings, FakeTools())
    try:
        with TestClient(create_app(settings, svc)) as client:
            payload = {"messages": [{"role": "user", "content": "Анализ повреждения панели"}], "chat_id": "chat", "message_id": "message", "stream": True}
            headers = {"Authorization": "Bearer " + settings.api_token, "X-User-Id": "alice"}
            assert client.post("/v1/chat/completions", json=payload).status_code == 401
            assert client.post("/v1/chat/completions", json=payload, headers={**headers, "X-User-Id": "outsider"}).status_code == 403
            response = client.post("/v1/chat/completions", json=payload, headers=headers)
            assert response.status_code == 200, response.text
            assert '"reasoning_content"' in response.text
            assert '"content"' in response.text
            assert response.text.endswith("data: [DONE]\n\n")
            chunks = [json.loads(x[6:]) for x in response.text.splitlines() if x.startswith("data: {")]
            job_id = chunks[0]["id"].removeprefix("chatcmpl-")
            assert client.get(f"/api/jobs/{job_id}", headers={**headers, "X-User-Id": "bob"}).status_code == 404
            calls = len(svc.tools.calls)
            again = client.post("/v1/chat/completions", json=payload, headers=headers)
            assert again.status_code == 200
            assert len(svc.tools.calls) == calls
            assert "reasoning_content" in again.text
    finally:
        svc.close()


def test_missing_chat_identity_fails(settings, service):
    with TestClient(create_app(settings, service)) as client:
        headers = {"Authorization": "Bearer " + settings.api_token, "X-User-Id": "alice"}
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}, headers=headers).status_code == 422


def test_config_requires_real_token(monkeypatch):
    monkeypatch.setenv("MRO_AGENT_API_TOKEN", "")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_only_internal_destinations(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))])
    with pytest.raises(ToolError, match="external_address_blocked"):
        validate_internal_url("http://example.com/api")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.2", 80))])
    validate_internal_url("http://internal-service/api")


def test_existing_search_wire_contract(settings):
    tools = Tools(settings)
    calls = []
    tools.kb = lambda method, path, user, payload: calls.append((path, payload)) or {}
    tools.search("панель", {"object": "панель", "task": "анализ", "identifiers": []}, "alice")
    assert calls[0][0] == "/api/similar-cases/search"
    assert calls[0][1]["limits"] == {"accepted": 5, "not_accepted": 5, "intermediate": 0}
    assert "retrieval_mode" not in calls[0][1]  # Never silently opt into legacy ranking.
    tools.close()


def test_http_reader_contract_and_authorization(settings, monkeypatch):
    import httpx
    monkeypatch.setattr("mro_agent.clients.validate_internal_url", lambda url: None)
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.path == "/api/case-facts":
            body = json.loads(request.content)
            assert body["case_id"] == "DEMO-1"
            return httpx.Response(200, json={"requested_case_id": "DEMO-1", "resolved_case_id": "DEMO-1", "resolution_method": "EXACT_INTERNAL_ID"})
        if request.url.path == "/api/cases/DEMO-1":
            return httpx.Response(200, json={"ok": True, "case": {"case_id": "DEMO-1", "documents": [{"document_id": "D1"}]}})
        if request.url.path == "/api/documents/D1":
            return httpx.Response(200, json={"ok": True, "document": {"document_id": "D1", "case_id": "DEMO-1", "chunks": []}})
        return httpx.Response(403, json={"private": "do not expose"})

    tools = Tools(settings)
    tools.http.client.close()
    tools.http.client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        resolved = tools.resolve("DEMO-1", "alice")
        docs = tools.case(resolved, "alice")["documents"]
        assert tools.document(docs[0]["document_id"], "alice")["case_id"] == "DEMO-1"
        assert all(r.headers["X-User-Id"] == "alice" for r in calls)
        with pytest.raises(ToolError, match="access_denied"):
            tools.document("forbidden", "alice")
    finally:
        tools.close()


def test_http_invalid_json_and_redirect_are_not_followed(monkeypatch):
    import httpx
    monkeypatch.setattr("mro_agent.clients.validate_internal_url", lambda url: None)
    http = JsonHTTP()
    http.client.close()
    http.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(302, headers={"Location": "https://outside.example"})))
    with pytest.raises(ToolError, match="invalid_http_response"):
        http.call("GET", "http://localhost/redirect")
    http.close()
    http.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not json")))
    with pytest.raises(ToolError, match="invalid_json"):
        http.call("GET", "http://localhost/json")
    http.close()


def test_three_unhelpful_answers_stop_research(service):
    first = run(service, "Нужен анализ")
    for i in range(1, 4):
        last = run(service, "Не знаю", f"m{i+1}")
    assert result(last)["status"] == "failed"
    assert "search" not in service.tools.calls


def test_attachment_warning_on_clarification(service):
    value = result(run(service, "Нужна оценка", attachments=True))
    assert value["status"] == "waiting"
    assert "Вложения" in value["content"]


def test_pipe_uses_original_prompt_and_preserves_reasoning(monkeypatch):
    import asyncio
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_pipe", Path(__file__).parents[1] / "openwebui" / "pipe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Response:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"reasoning_content":"Читаю документ"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"Результат"}}]}'
            yield "data: [DONE]"

    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def stream(self, method, url, **kwargs):
            calls.append(kwargs)
            return Response()

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    pipe = module.Pipe()
    pipe.valves.API_TOKEN = "service-token"
    async def collect(metadata):
        return [item async for item in pipe.pipe({"messages": [{"role": "user", "content": "WRAPPED SOURCES"}], "files": ["synthetic"]}, {"id": "alice"}, metadata)]
    output = asyncio.run(collect({"user_prompt": "исходная заявка", "chat_id": "chat", "message_id": "message"}))
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "исходная заявка"}]
    assert calls[0]["headers"]["X-User-Id"] == "alice"
    assert "reasoning_content" in output[0]["choices"][0]["delta"]
    asyncio.run(collect({"task": "title_generation"}))
    asyncio.run(collect({}))
    assert len(calls) == 1
