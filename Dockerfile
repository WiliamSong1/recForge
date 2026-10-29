FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY recforge ./recforge
RUN pip install --no-cache-dir .
RUN useradd --create-home recforge && mkdir /state && chown recforge /state
USER recforge
ENV RECFORGE_DATABASE=/state/profiles.db RECFORGE_ARTIFACT=/models
EXPOSE 8000
CMD ["uvicorn", "recforge.api:app", "--host", "0.0.0.0", "--port", "8000"]
