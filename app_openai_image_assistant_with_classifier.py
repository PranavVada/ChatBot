# app_openai_image_assistant_with_classifier.py
"""
Gradio image assistant with a model-based classifier (optional).

This version:
 - Keeps the provided default OPENAI_API_KEY per your request (warning: security risk).
 - Adds defensive OpenAI Responses parsing.
 - Falls back to heuristic classifier when network or key is unavailable.
 - Makes the startup image optional.
 - Improves requirements handling and README guidance.
 - Uses typing.Optional for Python 3.8+ compatibility.

USAGE:
 - Optionally set OPENAI_API_KEY env var to override the embedded key.
 - Place an image at `image.png` next to this script or start without one.
"""

import os
import re
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Any, Dict
from mimetypes import guess_type

from PIL import Image
import gradio as gr

# Optional OCR
try:
    import pytesseract
except Exception:
    pytesseract = None

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_assistant")

# ---------------- Config ----------------
# Per your request this ships with the key embedded.
# WARNING: embedding API keys in code is insecure. Prefer setting OPENAI_API_KEY in the environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
USE_MODEL_CLASSIFIER = True
SINGLE_IMAGE_PATH = Path("image.png")

# ---------------- Utilities ----------------
def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def _img_to_data_url(image_path: Path) -> Optional[str]:
    try:
        mime_type, _ = guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"
    except Exception as e:
        logger.info("Failed to make data URL: %s", e)
        return None

def ocr_image_to_text(image_path: Path) -> str:
    text = ""
    try:
        if pytesseract and image_path.exists():
            img = Image.open(str(image_path))
            text = normalize_ws(pytesseract.image_to_string(img) or "")
    except Exception as e:
        logger.info("OCR failed: %s", e)
    return text

# ---------------- Heuristics ----------------
_BAD_WORDS = {"fuck","f*ck","shit","bitch","asshole","bastard","dick","cunt","motherfucker","mf","bullshit","slut","whore","piss","bloody"}
_INTERIOR_KEYWORDS = {"interior","design","decor","room","living","bedroom","kitchen","bathroom","sofa","couch","chair","table","island","counter","backsplash","wardrobe","cabinet","shelf","storage","carpet","rug","tile","floor","flooring","paint","color","colour","palette","wall","ceiling","lighting","lamp"}
_IMAGE_DEICTICS = {"this","here","below","above","in this","in the image","in the photo","in the picture","in the design","in the room","as shown","shown here"}

def _contains_bad_words(q: str) -> bool:
    ql = q.lower()
    for w in _BAD_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", ql):
            return True
    return False

def _is_relevant_to_interior(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in _INTERIOR_KEYWORDS)

def _is_specific_to_image(q: str, last_mode: str, convo_pairs: list, caption_text: Optional[str]) -> bool:
    ql = q.lower()
    deictic_hit = any(phrase in ql for phrase in _IMAGE_DEICTICS)
    referential = any(x in ql for x in ["this","that","it","there","here"])
    object_hit = any(k in ql for k in _INTERIOR_KEYWORDS)
    caption_hit = False
    if caption_text:
        caption_hit = any(k in caption_text.lower() for k in _INTERIOR_KEYWORDS)
    if last_mode == "image":
        if len(ql.split()) <= 20 and (deictic_hit or referential or object_hit or caption_hit):
            return True
    if convo_pairs and len(ql.split()) <= 12 and (deictic_hit or referential):
        return True
    if deictic_hit:
        return True
    return object_hit and (referential or caption_hit)

def classify_question_manual(question: str, last_mode: str, convo_pairs: list, caption_text: Optional[str]) -> dict:
    q = (question or "").strip()
    if not q:
        return {"bad": False, "irrelevant": False, "general": False, "topic": None, "specific": True}
    if _contains_bad_words(q):
        return {"bad": True, "irrelevant": False, "general": False, "topic": None, "specific": False}
    if not _is_relevant_to_interior(q):
        return {"bad": False, "irrelevant": True, "general": False, "topic": None, "specific": False}
    specific = _is_specific_to_image(q, last_mode=last_mode, convo_pairs=convo_pairs, caption_text=caption_text)
    if specific:
        return {"bad": False, "irrelevant": False, "general": False, "topic": None, "specific": True}
    ql = q.lower()
    topic = None
    topic_map = {
        "paint": ["paint","colour","color","palette"],
        "lighting": ["light","lighting","lamp","pendant"],
        "flooring": ["floor","tile","laminate","wood","marble","granite"],
        "storage": ["storage","wardrobe","cabinet","shelf","drawers"],
        "kitchen": ["kitchen","island","counter","worktop","backsplash"],
        "bathroom": ["bathroom","toilet","shower","vanity"],
        "bedroom": ["bedroom","bed","wardrobe","nightstand"],
        "living": ["living","sofa","couch","coffee table"]
    }
    for t, kws in topic_map.items():
        if any(k in ql for k in kws):
            topic = t
            break
    return {"bad": False, "irrelevant": False, "general": True, "topic": topic, "specific": False}

# ---------------- OpenAI helpers ----------------
def _extract_response_text(resp: Any) -> Optional[str]:
    """Defensively extract text from Responses-like SDK objects or dicts."""
    try:
        # attribute style (some SDKs)
        if getattr(resp, "output_text", None):
            return resp.output_text
        out = getattr(resp, "output", None)
        if out:
            # try dict/list style
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                # if first is dict-like
                if isinstance(first, dict):
                    content = first.get("content") or first.get("data") or []
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and "text" in c:
                                return c["text"]
                            if isinstance(c, dict) and c.get("type") == "output_text" and "text" in c:
                                return c["text"]
                    # fallback: try first.get('text')
                    if "text" in first:
                        return first["text"]
        # dict-like response
        if isinstance(resp, dict):
            # traverse to find any 'text' fields
            def find_text(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        if k == "text" and isinstance(v, str):
                            return v
                        res = find_text(v)
                        if res:
                            return res
                elif isinstance(x, list):
                    for item in x:
                        res = find_text(item)
                        if res:
                            return res
                return None
            t = find_text(resp)
            if t:
                return t
        # last fallback: stringify
        s = str(resp)
        return s
    except Exception as e:
        logger.info("Failed to extract response text: %s", e)
        return None

def classify_with_model(question: str) -> Optional[dict]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "You are a classifier for user questions about interior design and images.\\n"
            "Classify the following user question and respond with JSON only (no extra text).\\n"
            "Return keys: bad (true/false), irrelevant (true/false), specific (true/false), topic (string|null).\\n\\n"
            f"Question: {question}"
        )
        resp = client.responses.create(
            model=CLASSIFIER_MODEL,
            input=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_output_tokens=200,
        )
        text = _extract_response_text(resp)
        if not text:
            return None
        # try to find JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            try:
                parsed = json.loads(text)
            except Exception:
                return None
        else:
            json_text = text[start:end+1]
            parsed = json.loads(json_text)
        bad = bool(parsed.get("bad", False))
        irrelevant = bool(parsed.get("irrelevant", False))
        specific = bool(parsed.get("specific", False))
        topic = parsed.get("topic") if parsed.get("topic") not in ("", None) else None
        return {"bad": bad, "irrelevant": irrelevant, "general": not specific and not irrelevant and not bad, "topic": topic, "specific": specific}
    except Exception as e:
        logger.info("Model classifier failed: %s", e)
        return None

def call_openai_with_context(question: str, llm_history: list, data_url: Optional[str], ocr_text: Optional[str], use_image: bool) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set — skipping remote call.")
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        msgs = [{"role":"system","content":"You are an expert interior designer who explains things simply to a layperson. Use short bullet points and concise answers (under ~150 words). When referencing the image, be specific and practical."}]
        if ocr_text:
            msgs.append({"role":"system","content":f"OCR text from the image:\\n{ocr_text}"})
        if llm_history:
            msgs.extend(llm_history[-20:])
        if use_image and data_url:
            user_content = [{"type":"input_text","text":question}, {"type":"input_image","image_url":data_url}]
            msgs.append({"role":"user","content":user_content})
        else:
            msgs.append({"role":"user","content":question})
        resp = client.responses.create(model=OPENAI_MODEL, input=msgs, temperature=0.3, max_output_tokens=400)
        text = _extract_response_text(resp)
        return normalize_ws(text) if text else None
    except Exception as e:
        logger.exception("OpenAI call error: %s", e)
        return None

# ---------------- Main answer flow ----------------
def answer_question(question: str, convo_pairs: list, llm_history: list, last_mode: str, data_url: Optional[str], ocr_text: Optional[str], caption_text: Optional[str]):
    q = (question or "").strip()
    if not q:
        return convo_pairs, llm_history, last_mode, gr.update(value=""), gr.update(value=convo_pairs)
    verdict = None
    if USE_MODEL_CLASSIFIER and OPENAI_API_KEY:
        try:
            cls = classify_with_model(q)
            if cls is not None:
                verdict = cls
        except Exception as e:
            logger.info("Classifier exception, falling back to manual: %s", e)
            verdict = None
    if verdict is None:
        verdict = classify_question_manual(q, last_mode=last_mode, convo_pairs=convo_pairs, caption_text=caption_text)
    if verdict["bad"]:
        answer = "Please avoid offensive language. I can help once it’s phrased politely."
        convo_pairs = convo_pairs + [(q, answer)]
        llm_history = llm_history + [{"role":"user","content":q}, {"role":"assistant","content":answer}]
        return convo_pairs, llm_history, last_mode, gr.update(value=""), gr.update(value=convo_pairs)
    if verdict["irrelevant"]:
        answer = "I can help with interior design only. Please ask about rooms, layouts, finishes, lighting, storage, etc."
        convo_pairs = convo_pairs + [(q, answer)]
        llm_history = llm_history + [{"role":"user","content":q}, {"role":"assistant","content":answer}]
        return convo_pairs, llm_history, last_mode, gr.update(value=""), gr.update(value=convo_pairs)
    use_image = bool(verdict.get("specific", False)) and bool(data_url)
    answer = None
    if OPENAI_API_KEY:
        try:
            answer = call_openai_with_context(q, llm_history, data_url if use_image else None, ocr_text, use_image)
        except Exception:
            answer = None
    if not answer:
        if verdict.get("general", False):
            topic = verdict.get("topic")
            answer = {
                "paint": "• Pick 2–3 tones max (one dominant, one accent).\\n• Test swatches; watch morning–night.\\n• Low-sheen hides flaws.",
                "lighting": "• Layer ambient + task + accent.\\n• 2700–3000K for cozy; ~4000K for tasks.\\n• Use dimmers.",
                "flooring": "• Match material to use (tiles for wet zones).\\n• Medium tones hide dust better than dark gloss.",
                "storage": "• Go vertical; full-height units.\\n• Deep drawers beat shelves for pots.",
                "kitchen": "• Keep work triangle short.\\n• Prefer drawers for bases; tall pantry.\\n• Under-cabinet lights.",
                "bathroom": "• Non-slip tiles; slope to drain.\\n• Ventilation is a must.",
                "bedroom": "• 750–900 mm beside the bed.\\n• 2700K lights; blackout curtains.",
                "living": "• Plan seating first; keep 900 mm walkways.\\n• Rug: front legs on, size to zone."
            }.get(topic) or ("• Start with function: list daily needs.\\n• Keep 2–3 base colors + one accent.\\n• Layer lighting.")
        else:
            caps = (caption_text or "").lower()
            ocrt = (ocr_text or "").lower()
            heuristics = []
            if any(k in caps for k in ("kitchen","island","counter","stove","hob")) or any(k in ocrt for k in ("kitchen","island")):
                heuristics.append("Kitchen: check the work triangle and leave 900–1200 mm clearance around islands; add under-cabinet task lights.")
            if any(k in caps for k in ("sofa","couch","rug","coffee table")) or any(k in ocrt for k in ("sofa","couch","rug")):
                heuristics.append("Seating: keep ~900 mm circulation; size rugs so front legs of furniture sit on them.")
            if not heuristics:
                heuristics = ["I see the image; consider lighting, storage, and durable finishes."]
            answer = "\\n".join(f"• {h}" for h in heuristics)
    convo_pairs = convo_pairs + [(q, answer)]
    llm_history = llm_history + [{"role":"user","content":q}, {"role":"assistant","content":answer}]
    last_mode = "image" if use_image else "general"
    return convo_pairs, llm_history, last_mode, gr.update(value=""), gr.update(value=convo_pairs)

# ---------------- App boot ----------------
def startup_load():
    # Do not crash if image missing; allow starting without one.
    if not SINGLE_IMAGE_PATH.exists():
        logger.info("Image not found at %s — starting without a fixed image. Place image.png later.", SINGLE_IMAGE_PATH)
        data_url = None
        ocr_text = ""
        caption_text = None
    else:
        data_url = _img_to_data_url(SINGLE_IMAGE_PATH)
        ocr_text = ocr_image_to_text(SINGLE_IMAGE_PATH)
        caption_text = None
    return str(SINGLE_IMAGE_PATH), [], [], "general", data_url, ocr_text, caption_text, gr.update(value=[])

with gr.Blocks(title="Ask About This Image — Classifier-enabled") as demo:
    gr.Markdown("### Ask About This Image — model-based classifier enabled if OPENAI_API_KEY is set.")
    with gr.Row():
        with gr.Column(scale=1):
            image_display = gr.Image(label="Fixed Image", type="filepath", interactive=False)
            gr.Markdown("*Place `image.png` next to this script (or change SINGLE_IMAGE_PATH).* ")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            qbox = gr.Textbox(label="Your question", placeholder="Type your question here…", autofocus=True)
            ask = gr.Button("Ask")
            clear = gr.Button("Clear conversation")

    state_convo_pairs = gr.State([])
    state_llm_history = gr.State([])
    state_last_mode = gr.State("general")
    state_data_url = gr.State("")
    state_ocr = gr.State("")
    state_caption = gr.State("")

    def _init():
        img_path, convo_pairs, llm_history, last_mode, data_url, ocr_text, caption, chat_update = startup_load()
        intro = ("Assistant ready. Ask about the image. If responses seem slow, set OPENAI_MODEL to a smaller Responses-capable model (e.g. 'gpt-4o-mini').")
        convo_pairs = [("","" + intro)]
        llm_history = [{"role":"assistant","content":intro}]
        return img_path, convo_pairs, llm_history, last_mode, data_url, ocr_text, caption, gr.update(value=convo_pairs)

    demo.load(_init, outputs=[image_display, state_convo_pairs, state_llm_history, state_last_mode, state_data_url, state_ocr, state_caption, chatbot])

    ask.click(fn=answer_question, inputs=[qbox, state_convo_pairs, state_llm_history, state_last_mode, state_data_url, state_ocr, state_caption], outputs=[state_convo_pairs, state_llm_history, state_last_mode, qbox, chatbot])

    def _clear():
        return [], [], "general", gr.update(value=""), gr.update(value=[])

    clear.click(_clear, outputs=[state_convo_pairs, state_llm_history, state_last_mode, qbox, chatbot])

if __name__ == "__main__":
    logger.info("Starting app. OPENAI_API_KEY set: %s. Model classifier enabled: %s", bool(OPENAI_API_KEY), USE_MODEL_CLASSIFIER)
    demo.launch(share=False)
