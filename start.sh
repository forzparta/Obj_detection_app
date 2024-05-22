#!/bin/bash
celery -A app.celery_app.app worker --loglevel=INFO &
poetry run uvicorn app.fast:app --host 0.0.0.0 --port 3000
