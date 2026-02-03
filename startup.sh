#!/bin/bash

# 1. Enter the virtual environment created by Azure/GitHub Actions
# Note: Azure often puts the environment in /home/site/wwwroot/antenv
if [ -d "antenv" ]; then
    source antenv/bin/activate
fi

# 2. Start the Web Server
# --bind: tells Gunicorn to listen on port 8000 (standard for Azure)
# --timeout: increased to 600 if your AI processing takes a long time
# app:app refers to the 'app' object inside 'app.py'
gunicorn --bind 0.0.0.0:8000 --timeout 600 app:app
