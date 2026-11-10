FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir pytest==8.3.5 \
    && useradd --create-home --uid 10001 runner

WORKDIR /workspace
USER runner

ENTRYPOINT ["timeout", "--kill-after=5s", "60s"]
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
