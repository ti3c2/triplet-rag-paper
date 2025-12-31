FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

RUN python -m pip install --no-cache-dir $(python -c 'import tomllib; print(*tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"])')

COPY . .