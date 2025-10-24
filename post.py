#!/usr/bin/env python3
import os, json, base64, textwrap, re
from datetime import datetime
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# -----------------------------
# CONFIGURATION
# -----------------------------
OUT_DIR = "out"
FONT_PATH = "assets/fonts/IMFellEnglishSC-Regular.ttf"
OVERLAY_TEXT_ON_IMAGE = True  # set False to post clean image without text overlay

# You can override these by setting GitHub env vars
THEME = os.getenv("THEME", "discipline, focus, perseverance")
TONE  = os.getenv("TONE",  "motivational, concise, modern")
STYLE = os.getenv("STYLE", "dark minimalist aesthetic, cinematic lighting")

GEMINI_KEY       = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "800"))  # large allowance
STABILITY_KEY    = os.getenv("STABILITY_API_KEY", "").strip()

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")
if not STABILITY_KEY:
    raise RuntimeError("Missing STABILITY_API_KEY")

# -----------------------------
# GEMINI (no fallbacks, fail-fast)
# -----------------------------
def _gemini_call(text: str, max_tokens: int, temperature: float = 0.5) -> str:
    """
    Single call to Gemini v1beta. Large maxOutputTokens; no retries or backups.
    Raises with a clear message if no text or MAX_TOKENS occurs.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json"
        }
    }
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:400]}")
    data = r.json()

    # Typical: candidates[0].content.parts[].text
    cand0 = (data.get("candidates") or [{}])[0]
    finish = cand0.get("finishReason")
    parts = cand0.get("content", {}).get("parts", [])
    txt = "".join(p.get("text", "") for p in parts).strip()

    if not txt:
        if finish == "MAX_TOKENS":
            raise RuntimeError(
                f"Gemini hit MAX_TOKENS before emitting text. "
                f"Increase GEMINI_MAX_TOKENS (currently {max_tokens}) or shorten the prompt.\n"
                f"Raw shape: {str(data)[:400]}"
            )
        raise RuntimeError(f"Gemini returned no text. Raw shape: {str(data)[:400]}")

    return txt

def _first_json_object(s: str) -> str | None:
    """Extract first well-formed {...} substring by brace counting."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _try_parse_single(obj):
    """Return (quote, reflection) if dict has both; else None."""
    if not isinstance(obj, dict):
        return None
    q = (obj.get("quote") or "").strip()
    r = (obj.get("reflection") or "").strip()
    if q and r:
        return q, r
    return None

def _coerce_to_single(json_text: str):
    """
    Accept a dict or [dict]; otherwise try to extract the first {...}.
    No further calls to Gemini; fail-fast if still invalid.
    """
    # Direct parse
    try:
        data = json.loads(json_text)
        if isinstance(data, list) and data:
            parsed = _try_parse_single(data[0])
            if parsed:
                return parsed
        elif isinstance(data, dict):
            parsed = _try_parse_single(data)
            if parsed:
                return parsed
    except json.JSONDecodeError:
        pass

    # Try first {...}
    frag = _first_json_object(json_text)
    if frag:
        data = json.loads(frag)  # if this fails, let it raise
        parsed = _try_parse_single(data)
        if parsed:
            return parsed

    # As a last structural check, confirm keys are present (still fail-fast)
    if '"quote"' in json_text and '"reflection"' in json_text:
        raise RuntimeError(
            "Gemini responded with text containing the keys but invalid JSON structure. "
            "Inspect the response or adjust the prompt to ensure valid JSON."
        )

    raise RuntimeError(
        "Gemini did not return a valid single JSON object with keys "
        '"quote" and "reflection".'
    )

def generate_verse_and_reflection():
    prompt = (
        f"Write ONE short Christian quote and ONE short reflection about {THEME}.\n"
        "Return ONLY a single JSON object (no arrays, no extra keys, no prose):\n"
        '{ "quote": "<≤16 words, typographic quotes like “...”>", '
        '"reflection": "<1 sentence, ≤36 words, no hashtags>" }'
    )

    raw = _gemini_call(prompt, max_tokens=GEMINI_MAX_TOKENS, temperature=0.5)
    quote, refl = _coerce_to_single(raw)

    # Guardrails: quotes and lengths (local, not a fallback)
    q = quote.strip()
    if not q.startswith(("“", "\"")):
        q = "“" + q.strip('“"').strip()
    if not q.endswith(("”", "\"")):
        q = q.rstrip('”" ').rstrip(".!,;:") + "”"
    if len(q.split()) > 16:
        q = " ".join(q.split()[:16]).rstrip(".!,;:") + "”"

    words = refl.split()
    if len(words) > 36:
        refl = " ".join(words[:36]).rstrip(".!,;:") + "."

    return q, refl

# -----------------------------
# STABILITY IMAGE GENERATION
# -----------------------------
def generate_image_bytes(prompt, width=1024, height=1024):
    """
    Uses Stability's v2beta Stable Image (Core) endpoint.
    Returns raw image bytes (JPEG). No base64 parsing needed.
    """
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {
        "Authorization": f"Bearer {STABILITY_KEY}",
        "Accept": "image/*",
        "Content-Type": "application/json",
    }
    # v2beta uses aspect ratio instead of width/height. Use 1:1 for Instagram.
    body = {
        "prompt": prompt,
        "output_format": "jpeg",      # "png" also allowed
        "aspect_ratio": "1:1",        # square image
        # Optional controls:
        # "negative_prompt": "",
        # "seed": 0,
        # "cfg_scale": 7,
        # "steps": 30,
    }

    r = requests.post(url, headers=headers, json=body, timeout=180)
    if r.status_code >= 400:
        # When the API returns JSON errors, surface them
        try:
            err = r.json()
        except Exception:
            err = r.text[:400]
        raise RuntimeError(f"Stability API error {r.status_code}: {err}")
    return r.content


# -----------------------------
# TEXT OVERLAY
# -----------------------------
def draw_centered_text(img, verse, reflection):
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font_verse = ImageFont.truetype(FONT_PATH, 64)
        font_reflection = ImageFont.truetype(FONT_PATH, 36)
    except Exception:
        font_verse = font_reflection = ImageFont.load_default()

    verse_wrapped = textwrap.fill(verse, width=28)
    reflection_wrapped = textwrap.fill(reflection, width=38)

    def _draw_line(line, y, font, fill):
        w, h = draw.textsize(line, font=font)
        x = (W - w) / 2
        draw.text((x+2, y+2), line, font=font, fill=(0,0,0))  # shadow
        draw.text((x, y), line, font=font, fill=fill)
        return h

    y = int(H * 0.16)
    for line in verse_wrapped.split("\n"):
        h = _draw_line(line, y, font_verse, (255, 255, 255))
        y += h + 8

    y += 40
    for line in reflection_wrapped.split("\n"):
        h = _draw_line(line, y, font_reflection, (235, 235, 235))
        y += h + 6

    return img

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Text via Gemini (large max tokens, no fallback)
    verse, reflection = generate_verse_and_reflection()

    # 2) Image via Stability
    img_bytes = generate_image_bytes(
        f"{THEME}, {TONE}, {STYLE}, instagram composition, detailed, high quality"
    )
    base_img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # 3) Optional overlay
    final_img = draw_centered_text(base_img, verse, reflection) if OVERLAY_TEXT_ON_IMAGE else base_img

    # 4) Save
    today = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"{today}.jpg"
    out_path = os.path.join(OUT_DIR, filename)
    final_img.save(out_path, "JPEG", quality=95)

    # 5) Caption + payload
    hashtags = "#Discipline #Focus #Perseverance #DailyMotivation"
    caption = f"{verse}\n\n{reflection}\n\n{hashtags}"

    repo = os.getenv("GITHUB_REPOSITORY", "") or "owner/repo"
    image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"

    payload = {"image_url": image_url, "caption": caption}
    with open("post_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {out_path}\n{caption}")
