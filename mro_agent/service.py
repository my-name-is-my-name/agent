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
        followup_kind = None
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
                    kind = self.classify_followup(values, request_text)
                    followup_kind = kind
                    if kind == "question":
                        self.answer_question(job, values, request_text)
                        return
                    if kind == "new_request":
                        if values.get("pending_followup"):
                            graph.update_state(config, {"pending_followup": None})
                            values = dict(graph.get_state(config).values)
                        self.finish_routing_response(
                            job, values, kind,
                            "Это сообщение описывает отдельную заявку. Начните новую оценку в новом чате или контексте; текущая заявка не изменена.",
                        )
                        return
                    if kind == "ambiguous":
                        pending = values.get("pending_followup") or request_text
                        if not values.get("pending_followup"):
                            graph.update_state(config, {"pending_followup": pending})
                            values = dict(graph.get_state(config).values)
                        current = self.current_assessment_label(values)
                        self.finish_routing_response(
                            job, values, kind,
                            f"«{plain(pending[:240])}» относится к текущей заявке {current} или это отдельная заявка?",
                        )
                        return
                    addition = values.get("pending_followup") or request_text
                    request_text = values["request"] + "\nДополнение пользователя: " + addition
                    if len(request_text) > 48000:
                        raise ToolError("request_history_limit")
                initial = {
                    "request": request_text, "user_id": job["user_id"], "job_id": job_id,
                    "profile": {}, "questions": [], "rounds": 0, "candidates": [], "selected": [],
                    "evidence": [], "claims": [], "proposal": {}, "warnings": [], "content": "",
                    "status": "researching", "attachments_present": payload["attachments_present"],
                    "pending_followup": None,
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
            result = {"assessment_id": job["assessment_id"], "status": state.get("status"), "content": state.get("content") or render(state), "state": state}
            if followup_kind:
                result["followup_kind"] = followup_kind
            self.store.finish(job_id, result)
        except ToolError as exc:
            self.fail(job, exc.code)
        except Exception:
            self.fail(job, "internal_error")

    def classify_followup(self, state, message):
        intent = self.tools.llm(
            "Классифицируй новое сообщение в существующей MRO-оценке. "
            "question — вопрос о текущей оценке или уже полученных результатах; "
            "correction — пользователь явно добавляет, меняет, исправляет или уточняет сведения именно текущей заявки; "
            "new_request — пользователь явно обозначает другую/новую заявку либо описывает явно отдельный инженерный случай; "
            "ambiguous — небезопасно определять, относится ли инженерное сообщение к текущей заявке. "
            "Разные дефекты, места, детали или ВС нельзя объединять только потому, что сообщения находятся в одном чате. "
            "Если pending_followup задан, message — ответ пользователя на вопрос о принадлежности pending_followup: "
            "верни correction при подтверждении текущей заявки и new_request при подтверждении отдельной заявки. "
            "При сомнении выбирай ambiguous. Не считай короткий инженерный фрагмент correction без явной связи с текущей заявкой.",
            {
                "request": state.get("request"), "profile": state.get("profile", {}),
                "message": message, "pending_followup": state.get("pending_followup"),
            },
            Followup,
        )
        return intent["kind"]

    def current_assessment_label(self, state):
        profile = state.get("profile") or {}
        identifiers = profile.get("identifiers") or []
        if identifiers:
            return "по " + ", ".join(plain(str(value)) for value in identifiers[:3])
        object_name = str(profile.get("object") or "").strip()
        if object_name:
            return "по объекту «" + plain(object_name[:160]) + "»"
        return "по сохранённым исходным данным"

    def finish_routing_response(self, job, state, kind, content):
        self.store.event(job["id"], "followup", "completed", f"Сообщение классифицировано как {kind}; текущая заявка не изменена.")
        self.store.finish(job["id"], {
            "assessment_id": job["assessment_id"], "status": "ready", "content": content,
            "state": state, "followup_kind": kind,
        })

    def answer_question(self, job, state, question):
        self.store.event(job["id"], "question", "started", "Отвечаю по сохранённому состоянию оценки; исходные данные заявки не изменяю.")
        content = self.grounded_state_answer(state, question)
        if json.loads(job["payload"]).get("attachments_present"):
            content += "\n\nВложения не прочитаны: версия 0.1 отвечает только по тексту и ранее изученным источникам."
        self.store.event(job["id"], "question", "completed", "Ответ по сохранённым материалам подготовлен.")
        self.store.finish(job["id"], {"assessment_id": job["assessment_id"], "status": "ready", "content": content, "state": state, "followup_kind": "question"})

    def grounded_state_answer(self, state, question):
        sources = list(state.get("evidence", []))
        structured = {
            "profile": state.get("profile", {}), "candidates": state.get("candidates", []),
            "selected": state.get("selected", []), "proposal": state.get("proposal", {}),
            "warnings": state.get("warnings", []),
            "evidence": [{key: value for key, value in item.items() if key != "text"} for item in state.get("evidence", [])],
        }
        for key, value in structured.items():
            sources.append({"id": "STATE-" + key, "text": json.dumps(value, ensure_ascii=False, sort_keys=True)})
        result = self.tools.llm(
            "Ответь на вопрос по переданным сохранённым источникам. Каждый вывод должен содержать evidence_id и точную цитату. "
            "Источники STATE-* — метаданные работы агента, а не доказательство содержания документов. "
            "saved_claims можно использовать только как указатели: утверждение о содержании документа должно ссылаться на исходный E*-источник и его точную цитату. "
            "Не оценивай часы или выполнимость.",
            {"question": question, "sources": sources, "saved_claims": state.get("claims", [])}, Extraction,
        )
        evidence = {e["id"]: e for e in sources}
        lines = []
        for claim in result["claims"]:
            source = evidence.get(claim["evidence_id"])
            if source and claim["quote"] in source["text"]:
                if claim["evidence_id"].startswith("STATE-"):
                    lines.append(f"- {plain(claim['text'])} (по сохранённым метаданным оценки).")
                else:
                    verdict = self.tools.llm("Подтверждается ли утверждение цитатой с учётом контекста?", {"claim": claim, "context": source["text"]}, Verdict)
                    if verdict["supported"]:
                        lines.append(f"- {plain(claim['text'])} [{claim['evidence_id']}]. Цитата: «{plain(claim['quote'])}»")
        if state.get("candidates"):
            lines.append("- Найденные поиском кандидаты сами по себе не подтверждают техническую применимость.")
            if not state.get("evidence"):
                lines.append("- Документы кандидатов не были успешно прочитаны; техническая применимость не подтверждена.")
        return "Ответ по сохранённым материалам:\n\n" + ("\n".join(lines) if lines else "Недостаточно данных в сохранённом состоянии оценки.")

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
