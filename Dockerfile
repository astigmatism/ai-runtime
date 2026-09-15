FROM docker:29-cli@sha256:eccaacfeed644c7de222ff047483568cb988dde95476fbaaf10ea2d04921bb66 AS docker_cli
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker_cli /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
COPY --from=docker_cli /usr/local/libexec/docker/cli-plugins/docker-buildx /usr/local/lib/docker/cli-plugins/docker-buildx
ARG SOURCE_REVISION=development
LABEL org.opencontainers.image.source="https://github.com/astigmatism/local-ai-runtime" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.title="AI Runtime"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY runtime /app/runtime
COPY config /app/config
COPY tests /app/tests
RUN printf '%s\n' "$SOURCE_REVISION" > /app/REVISION
USER 1000:1000
CMD ["python3", "-m", "runtime", "serve"]
