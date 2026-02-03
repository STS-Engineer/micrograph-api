#!/bin/bash

# We do not need to manually activate antenv. 
# Azure's Python environment handles activation automatically 
# when SCM_DO_BUILD_DURING_DEPLOYMENT is set to true.

echo "Starting Gunicorn server..."
gunicorn --bind 0.0.0.0:8000 --timeout 600 app:app
