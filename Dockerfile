FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY mro_agent ./mro_agent
RUN pip install --no-cache-dir -c requirements.lock . && useradd --uid 10001 --create-home agent && mkdir /app/runtime && chown agent /app/runtime
USER agent
EXPOSE 8140
CMD ["uvicorn", "mro_agent.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8140", "--workers", "1"]
