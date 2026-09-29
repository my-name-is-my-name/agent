"""
title: MRO Assessment Agent
version: 0.1.0
description: Local MRO agent with persistent chat state and streamed progress.
requirements: httpx
"""
import json

import httpx
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        AGENT_URL: str = Field(default="http://mro-assessment-agent:8140")
        API_TOKEN: str = Field(default="", description="Agent service token (administrator only)")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": "mro-assessment-agent", "name": "MRO Assessment Agent"}]

    async def pipe(self, body: dict, __user__: dict, __metadata__: dict | None = None, __task__: str | None = None):
        metadata = __metadata__ or {}
        if __task__ or metadata.get("task"):
            # WebUI may call the selected model for titles/tags/query generation.
            # These calls must never create or resume an assessment.
            yield {"choices": [{"delta": {"content": "MRO Assessment"}}]}
            return
        user_id = str((__user__ or {}).get("id") or "")
        chat_id = str(metadata.get("chat_id") or "")
        message_id = str(metadata.get("message_id") or "")
        if not user_id or not chat_id or not message_id:
            yield {"choices": [{"delta": {"content": "Не получены идентификаторы пользователя, чата или сообщения. Откройте сохранённый обычный чат; оценка не создана."}}]}
            return
        # user_prompt is the documented original text before WebUI citation wrapping.
        messages = body.get("messages", [])
        has_media = any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") != "text" for p in m["content"]) for m in messages if m.get("role") == "user")
        if isinstance(metadata.get("user_prompt"), str):
            messages = [{"role": "user", "content": metadata["user_prompt"]}]
        elif body.get("files") or metadata.get("files"):
            yield {"choices": [{"delta": {"content": "В этой версии Open WebUI не передан исходный текст отдельно от вложений. Пришлите заявку текстом без вложений."}}]}
            return
        messages = next(([m] for m in reversed(messages) if m.get("role") == "user"), [])
        payload = {
            "model": "mro-assessment-agent", "messages": [{"role": m["role"], "content": m.get("content", "")} for m in messages if m.get("role") in ("user", "assistant")],
            "chat_id": chat_id, "message_id": message_id, "stream": True,
            "attachments_present": bool(body.get("files") or metadata.get("files") or has_media),
        }
        headers = {"Authorization": "Bearer " + self.valves.API_TOKEN, "X-User-Id": user_id}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=120), trust_env=False, follow_redirects=False) as client:
                async with client.stream("POST", self.valves.AGENT_URL.rstrip("/") + "/v1/chat/completions", headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        yield {"choices": [{"delta": {"content": f"Агент вернул HTTP {response.status_code}. Проверьте доступ или дождитесь завершения текущей оценки."}}]}
                        return
                    async for line in response.aiter_lines():
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            yield json.loads(line[6:])
        except (httpx.HTTPError, ValueError):
            yield {"choices": [{"delta": {"content": "Соединение с агентом прервано. Карточка сохраняется на сервере; повторите обращение после восстановления связи."}}]}
