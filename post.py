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

GEMINI_KEY     = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
STABILITY_KEY  = os.getenv("STABILITY_API_KEY", "").strip()

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")
if not STABILITY_KEY:
    raise RuntimeError("Missing STABILITY_API_KEY")

# -----------------------------
# GEMINI
# -----------------------------
def _gemini_call(text, max_tokens=160, temperature=0.6):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json"
        },
        # relax safety to avoid “blocked” for religious content
        "safetySettings": [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ],
    }
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    data = r.json()
    # typical path: candidates[0].content.parts[0].text
    try:
        parts = data["candidates"][0]["content"].get("parts", [])
        txt = "".join(p.get("text", "") for p in parts).strip()
        if not txt:
            raise KeyError("empty text")
        return txt
    except Exception:
        raise RuntimeError(f"Unexpected Gemini response: {str(data)[:400]}")

def _first_json_object(s: str) -> str | None:
    """
    Extract the first JSON object substring from s using a brace counter.
    Returns the substring or None if not found/unterminated.
    """
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
    return None  # unterminated

def _try_parse_single(obj) -> tuple[str, str] | None:
    """Return (quote, reflection) or None."""
    if not isinstance(obj, dict):
        return None
    q = (obj.get("quote") or "").strip()
    r = (obj.get("reflection") or "").strip()
    if q and r:
        return q, r
    return None

def _coerce_to_single(json_text: str) -> tuple[str, str]:
    """
    Accepts: a JSON dict or an array of dicts.
    If malformed, tries to extract first {...}. If still broken, raises.
    """
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

    # Try to grab the first object with a brace counter
    frag = _first_json_object(json_text)
    if frag:
        try:
            data = json.loads(frag)
            parsed = _try_parse_single(data)
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # As a last resort, try a very loose regex for keys and manually rebuild
    q_match = re.search(r'"quote"\s*:\s*"(.*?)"', json_text, re.DOTALL)
    r_match = re.search(r'"reflection"\s*:\s*"(.*?)"', json_text, re.DOTALL)
    if q_match and r_match:
        q = q_match.group(1).strip()
        r = r_match.group(1).strip()
        return q, r

    raise RuntimeError("Could not coerce Gemini output to a single JSON object.")

def _repair_with_gemini(bad_text: str) -> tuple[str, str]:
    """
    Ask Gemini to sanitize malformed/partial output into valid JSON object.
    """
    repair_prompt = (
        "Fix the following into a VALID JSON object with exactly these keys:\n"
        '{"quote": "<string>", "reflection": "<string>"}\n'
        "If text is truncated, complete it succinctly. No arrays. Return ONLY JSON.\n\n"
        f"INPUT:\n{bad_text}"
    )
    fixed = _gemini_call(repair_prompt, max_tokens=120, temperature=0.2)
    return _coerce_to_single(fixed)

def generate_verse_and_reflection():
    base_prompt = (
        f"Write a SINGLE short Christian quote and a short reflection about {THEME}.\n"
        "Output ONLY one JSON object (no arrays, no extra text):\n"
        '{ "quote": "<≤16 words, in typographic quotes like “...”>",\n'
        '  "reflection": "<1–2 sentences, ≤40 words, no hashtags>" }\n'
        "Rules:\n"
        "- Use typographic quotes for the quote itself.\n"
        "- The reflection is Instagram-friendly and concise.\n"
        "- No additional keys. No explanations."
    )

    raw = _gemini_call(base_prompt, max_tokens=160, temperature=0.55)

    # Parse or repair
    try:
        quote, refl = _coerce_to_single(raw)
    except Exception:
        quote, refl = _repair_with_gemini(raw)

    # Guardrails: ensure proper wrapping and length caps
    q = quote.strip()
    if not q.startswith(("“", "\"")):
        q = "“" + q.strip('“"').strip()
    if not q.endswith(("”", "\"")):
        q = q.rstrip('”" ').rstrip(".!,;:") + "”"
    if len(q.split()) > 16:
        q = " ".join(q.split()[:16]).rstrip(".!,;:") + "”"

    words = refl.split()
    if len(words) > 40:
        refl = " ".join(words[:40]).rstrip(".!,;:") + "."

    return q, refl

# -----------------------------
# STABILITY IMAGE GENERATION
# -----------------------------
def generate_image_bytes(prompt, width=1024, height=1024):
    url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
    headers = {"Authorization": f"Bearer {STABILITY_KEY}"}
    body = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "samples": 1
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    j = r.json()
    if "artifacts" not in j or not j["artifacts"]:
        raise RuntimeError(f"Stability API returned no image: {str(j)[:200]}")
    return base64.b64decode(j["artifacts"][0]["base64"])

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

    # subtle shadow for legibility
    def _draw_line(line, y, font, fill):
        w, h = draw.textsize(line, font=font)
        x = (W - w) / 2
        # shadow
        draw.text((x+2, y+2), line, font=font, fill=(0,0,0))
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

    # 1) Text via Gemini (robust)
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

    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not repo:
        # Actions always sets this; but fall back for local runs
        repo = "owner/repo"
    image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"

    payload = {"image_url": image_url, "caption": caption}
    with open("post_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {out_path}\n{caption}")
