# injectkit — containerized prompt-injection scanner.
#
# DEFENSIVE / AUTHORIZED USE ONLY. Scan endpoints you own or are authorized to
# test. This image bundles injectkit (core + anthropic + mcp extras) so it can
# run a scan in CI or anywhere Docker runs, e.g.:
#
#   docker build -t injectkit .
#   docker run --rm -e ANTHROPIC_API_KEY -v "$PWD:/work" -w /work injectkit \
#     scan --target mock --format sarif --out injectkit-results.sarif
#
# The GitHub Action (action.yml) is a composite action that runs injectkit
# directly on the runner; this Dockerfile is the portable, runner-agnostic way
# to get the same scan, and powers `docker run` usage outside Actions.
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="injectkit" \
      org.opencontainers.image.description="Red-team your own LLM apps for prompt injection. Authorized-use only." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/Dukotah/injectkit"

# No bytecode files, unbuffered stdout (so logs stream in CI).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy only what's needed to install the package first, for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY injectkit ./injectkit

# Install injectkit with all optional adapters (anthropic + mcp) so any target
# kind works out of the box. The corpus YAML ships inside the wheel.
RUN pip install --upgrade pip \
 && pip install ".[all]"

# Copy the Action entrypoint so the image can be used as a Docker action too.
COPY entrypoint.sh /usr/local/bin/injectkit-entrypoint
RUN chmod +x /usr/local/bin/injectkit-entrypoint

# Default to a non-root user for safety; CI mounts are world-readable.
RUN useradd --create-home --uid 1001 injectkit
USER injectkit

# `docker run injectkit <args>` -> `injectkit <args>` (e.g. `scan ...`).
ENTRYPOINT ["injectkit"]
CMD ["--help"]
