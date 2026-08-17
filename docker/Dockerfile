# syntax=docker/dockerfile:1
#
# TestAgent runtime image.
#
# Build (run from repo root):
#   docker build -f docker/Dockerfile -t testagent .
#
# Run — the entrypoint is `testagent`, so any CLI args are passed straight
# through. Results are written under $OUTPUT_DIR (default /app/output); mount
# a host directory there with -v so artifacts land on your host:
#
#   # 1) Recommended: mount the current dir for inputs AND output
#   docker run --rm -v "$PWD":/work -w /work \
#       -e OPENAI_API_KEY="$OPENAI_API_KEY" \
#       testagent generate-tests -r requirements.md -s swagger.json \
#       -o output/testcases.json
#
#   # 2) Mount only the output dir (inputs must be URLs or already in image):
#   docker run --rm -v "$PWD/output":/app/output \
#       -e OPENAI_API_KEY="$OPENAI_API_KEY" \
#       testagent generate-tests -s https://example.com/openapi.json \
#       -o output/testcases.json
#
#   # 3) Run the web GUI (embeddable via iframe):
#   docker run --rm -p 8000:8000 -e OPENAI_API_KEY="$OPENAI_API_KEY" \
#       testagent serve --host 0.0.0.0 --port 8000
#
# All settings can be overridden with -e flags (see .env.example).

FROM python:3.14-slim AS runtime

# uv: a single static binary, copied from the official image (no pip needed).
# UV_LINK_MODE=copy avoids bind-mount issues on some filesystems;
# UV_COMPILE_BYTECODE=1 precompiles .pyc for faster cold start.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Keep Python output unbuffered, disable pyc clutter, and default the output
# directory to the conventional mount point.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE_DIR=1 \
    OUTPUT_DIR=/app/output

WORKDIR /app

# Install the package with document-parsing + web-GUI extras. These deps
# (python-docx, pdfplumber, beautifulsoup4, markdownify, python-pptx, fastapi,
# uvicorn) all ship manylinux/macos wheels, so no system build tools are
# required — keeping the image small. uv installs ~10x faster than pip.
COPY pyproject.toml README.md ./
COPY testagent/ testagent/
COPY templates/ templates/

RUN uv pip install --system ".[documents,web]"

# Pre-create the output mount point so a read-only-ish bind still works and
# ownership is correct before we drop privileges.
RUN mkdir -p /app/output \
    && useradd --create-home --uid 1001 testagent \
    && chown -R testagent:testagent /app

USER testagent

# `testagent` is the console script declared in pyproject [project.scripts].
# Any `docker run testagent <args>` is forwarded to it.
ENTRYPOINT ["testagent"]
CMD ["--help"]
