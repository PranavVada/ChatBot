Architectural Chatbot

This repo is an updated version of your Gradio image assistant. **Per your request the original API key is embedded in the script.**
**Warning:** embedding API keys in source code is insecure. Prefer setting `OPENAI_API_KEY` in the environment or using a `.env` file.

## What's changed
- Defensive parsing of Responses API outputs.
- Graceful fallback to heuristics if API or key is unavailable.
- Startup no longer crashes if `image.png` is missing.
- Better requirements and README hints.

## Quickstart
1. (Optional) Create a virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place `image.png` next to the script (or run without it).

3. Run:
```bash
python app_openai_image_assistant_with_classifier.py
```

To override the embedded key, set the env var:
```bash
export OPENAI_API_KEY="your_key_here"
```
