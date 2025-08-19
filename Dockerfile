FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .[dev]
COPY . /app
CMD ["python", "-m", "bybitbot.bot"]
