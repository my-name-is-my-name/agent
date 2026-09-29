import fcntl
import json
import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langsmith import tracing_context

from .clients import ToolError, Tools
from .models import Extraction, Followup, Verdict
from .storage import Store
from .workflow import Workflow, plain, render


class Service:
    def __init__(self, settings, tools=None, start_worker=True):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lockfile = (settings.data_dir / "worker.lock").open("a")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lockfile.close()
            raise RuntimeError("Only one process may use this runtime directory") from None
        self.store = Store(settings.data_dir / "assessments.sqlite3")
        self.connection = sqlite3.connect(settings.data_dir / "checkpoints.sqlite3", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.tools = tools or Tools(settings)
        self.workflow = Workflow(settings, self.tools, self.store, self.checkpointer)
        self.stopping = threading.Event()
        self.wakeup = threading.Event()
        self.store.recover()
        self.worker = threading.Thread(target=self.work, name="mro-agent-worker", daemon=True)
        if start_worker:
            self.worker.start()

    def submit(self, user_id, request):
        job = self.store.submit(user_id, request.chat_id, request.message_id, {
            "request": request.request, "attachments_present": request.attachments_present,
        })
        self.wakeup.set()
        return job

    def work(self):
        while not self.stopping.is_set():
            job_id = self.store.claim_next()
            if job_id:
                self.process(job_id)
            else:
                self.wakeup.wait(0.5)
                self.wakeup.clear()

    def process(self, job_id):
        # Never inherit cloud tracing from an administrator's shell environment.
        with tracing_context(enabled=False):
            self._process(job_id)

    def _process(self, job_id):
        job = self.store.job(job_id)
        payload = json.loads(job["payload"])
        config = {"configurable": {"thread_id": job["assessment_id"], "job_id": job_id}, "recursion_limit": 30}
        graph = self.workflow.graph
        self.store.event(job_id, "queue", "started", "Начинаю обработку сообщения в сохранённой карточке заявки.")
        try:
            snapshot = graph.get_state(config)
            values = snapshot.values or {}
            interrupted = any(task.interrupts for task in snapshot.tasks)
            if snapshot.next:
                if interrupted and values.get("job_id") == job_id:
                    # Crash after persisting the interrupt but before persisting the API result.
                    pass
                elif interrupted:
                    graph.invoke(Command(resume={"text": payload["request"], "attachments_present": payload["attachments_present"]}), config)
                else:
                    graph.invoke(None, config)
            elif values.get("job_id") == job_id:
                pass  # Graph completed before process restart; publish that same result.
            else:
                request_text = payload["request"]
                if values:
                    intent = self.tools.llm("Классифицируй новое сообщение: question — вопрос о сохранённом результате; correction — дополнение или исправление исходных данных заявки. Не меняй исходную заявку из-за вопроса.", {"request": values.get("request"), "message": request_text}, Followup)
                    if intent["kind"] == "question":
                        self.answer_question(job, values, request_text)
                        return
                    request_text = values["request"] + "\nДополнение пользователя: " + request_text
                    if len(request_text) > 48000:
                        raise ToolError("request_history_limit")
                initial = {
                    "request": request_text, "user_id": job["user_id"], "job_id": job_id,
                    "profile": {}, "questions": [], "rounds": 0, "candidates": [], "selected": [],
                    "evidence": [], "claims": [], "proposal": {}, "warnings": [], "content": "",
                    "status": "researching", "attachments_present": payload["attachments_present"],
                }
                graph.invoke(initial, config)
            snapshot = graph.get_state(config)
            state = dict(snapshot.values)
            waiting = any(task.interrupts for task in snapshot.tasks)
            if waiting:
                self.store.event(job_id, "clarify", "waiting", "Ожидаю ответа пользователя на уточняющие вопросы.")
                state["status"] = "waiting"
                state["content"] = "Для продолжения уточните:\n\n" + "\n".join("- " + plain(q) for q in state.get("questions", []))
                if state.get("attachments_present"):
                    state["content"] += "\n\nВложения версии 0.1 не прочитаны. Укажите необходимые сведения текстом."
            self.store.finish(job_id, {"assessment_id": job["assessment_id"], "status": state.get("status"), "content": state.get("content") or render(state), "state": state})
        except ToolError as exc:
            self.fail(job, exc.code)
        except Exception:
            self.fail(job, "internal_error")

    def answer_question(self, job, state, question):
        self.store.event(job["id"], "question", "started", "Отвечаю по сохранённым источникам; исходные данные заявки не изменяю.")
        result = self.tools.llm("Ответь на вопрос только утверждениями с точными цитатами из переданных источников. Если ответа нет, верни пустой claims. Не оценивай часы и выполнимость.", {"question": question, "sources": state.get("evidence", [])}, Extraction)
        evidence = {e["id"]: e for e in state.get("evidence", [])}
        lines = []
        for claim in result["claims"]:
            source = evidence.get(claim["evidence_id"])
            if source and claim["quote"] in source["text"]:
                verdict = self.tools.llm("Подтверждается ли утверждение цитатой с учётом контекста?", {"claim": claim, "context": source["text"]}, Verdict)
                if verdict["supported"]:
                    lines.append(f"- {plain(claim['text'])} [{claim['evidence_id']}]. Цитата: «{plain(claim['quote'])}»")
        content = "Ответ по сохранённым источникам:\n\n" + ("\n".join(lines) if lines else "Недостаточно данных в изученных источниках.")
        if json.loads(job["payload"]).get("attachments_present"):
            content += "\n\nВложения не прочитаны: версия 0.1 отвечает только по тексту и ранее изученным источникам."
        self.store.event(job["id"], "question", "completed", "Ответ по сохранённым материалам подготовлен.")
        self.store.finish(job["id"], {"assessment_id": job["assessment_id"], "status": "ready", "content": content, "state": state})

    def fail(self, job, code):
        self.store.event(job["id"], "error", "error", "Оценка не завершена: " + code)
        self.store.finish(job["id"], {"assessment_id": job["assessment_id"], "status": "failed", "content": "Не удалось завершить оценку: " + code + ". Данных для решения недостаточно. Карточка сохранена; повторное обращение продолжит незавершённый этап."}, failed=True)

    def close(self):
        self.stopping.set()
        self.wakeup.set()
        if self.worker.is_alive():
            self.worker.join(timeout=5)
        if not self.worker.is_alive():
            self.tools.close()
            self.connection.close()
            self.lockfile.close()
