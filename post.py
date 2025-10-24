#!/usr/bin/env python3
import os, json, base64, textwrap, re, time
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
def _gemini_call(text, max_tokens=160, temperature=0.55, retries=2):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json"
        }
        # NOTE: safetySettings removed to avoid INVALID_ARGUMENT errors
    }

    for attempt in range(retries + 1):
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code == 429 and attempt < retries:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            parts = data["candidates"][0]["content"].get("parts", [])
            txt = "".join(p.get("text", "") for p in parts).strip()
            if not txt:
                raise KeyError("empty text")
            return txt
        except Exception:
            # try once more with fewer tokens if shape is odd
            if attempt < retries:
                payload["generationConfig"]["maxOutputTokens"] = max(120, int(max_tokens * 0.75))
                continue
            raise RuntimeError(f"Unexpected Gemini response: {str(data)[:400]}")

def _first_json_object(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth, i = 0, start
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
        i += 1
    return None

def _try_parse_single(obj):
    if not isinstance(obj, dict):
        return None
    q = (obj.get("quote") or "").strip()
    r = (obj.get("reflection") or "").strip()
    if q and r:
        return q, r
    return None

def _coerce_to_single(json_text: str):
    # Accept dict or array-of-dicts, else try extraction
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

    frag = _first_json_object(json_text)
    if frag:
        try:
            data = json.loads(frag)
            parsed = _try_parse_single(data)
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    q_match = re.search(r'"quote"\s*:\s*"(.*?)"', json_text, re.DOTALL)
    r_match = re.search(r'"reflection"\s*:\s*"(.*?)"', json_text, re.DOTALL)
    if q_match and r_match:
        return q_match.group(1).strip(), r_match.group(1).strip()

    raise RuntimeError("Could not coerce Gemini output to a single JSON object.")

def _repair_with_gemini(bad_text: str):
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
        "- Use typographic quotes for the quote text.\n"
        "- The reflection is concise and Instagram-friendly.\n"
        "- No additional keys."
    )

    raw = _gemini_call(base_prompt, max_tokens=160, temperature=0.55)

    try:
        quote, refl = _coerce_to_single(raw)
    except Exception:
        quote, refl = _repair_with_gemini(raw)

    # Guardrails: quotes and lengths
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
    headers = {
        "Authorization": f"Bearer {STABILITY_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
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

    repo = os.getenv("GITHUB_REPOSITORY", "") or "owner/repo"
    image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"

    payload = {"image_url": image_url, "caption": caption}
    with open("post_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {out_path}\n{caption}")
