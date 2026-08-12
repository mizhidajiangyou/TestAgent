FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY testagent/ testagent/
COPY templates/ templates/

RUN pip install --no-cache-dir .

ENTRYPOINT ["testagent"]
CMD ["--help"]
