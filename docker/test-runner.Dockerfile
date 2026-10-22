FROM python:3.11-slim

ARG PYTEST_VERSION=8.4.2

RUN python -m pip install --no-cache-dir "pytest==${PYTEST_VERSION}" \
    && groupadd --gid 10001 coderepair \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin coderepair

USER 10001:10001
WORKDIR /workspace

# CodeRepair supplies an argv directly. No shell entrypoint is involved.
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-q"]

LABEL org.opencontainers.image.title="CodeRepair pytest sandbox" \
      org.opencontainers.image.description="Minimal offline Python 3.11 + pytest test runner"

