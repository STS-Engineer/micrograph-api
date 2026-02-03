#!/bin/bash
set -e

echo "Starting Gunicorn server on PORT=${PORT:-8000}..."
exec gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 600 app:app
