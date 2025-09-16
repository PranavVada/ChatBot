#!/usr/bin/env bash
set -e
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
pip install -r requirements.txt
python app_openai_image_assistant_with_classifier.py
