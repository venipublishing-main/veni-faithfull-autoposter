import os, json, base64, textwrap
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

GEMINI_KEY     = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
STABILITY_KEY  = os.getenv("STABILITY_API_KEY")

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")
if not STABILITY_KEY:
    raise RuntimeError("Missing STABILITY_API_KEY")

# -----------------------------
# GEMINI TEXT GENERATION (JSON)
# -----------------------------
def gemini_generate(prompt: str) -> str:
    """
    Calls Google GenAI (Gemini) and returns the first text part.
    We request JSON via responseMimeType to make parsing robust.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 320,
            "responseMimeType": "application/json"
        }
    }
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        parts = data["candidates"][0]["content"].get("parts", [])
        if not parts or "text" not in parts[0]:
            raise KeyError("No text parts in response")
        return parts[0]["text"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected Gemini response shape: {str(data)[:300]}")

def generate_verse_and_reflection():
    prompt = (
        f"Generate TWO short pieces of text about {THEME} and return ONLY this JSON object:\n"
        '{ "quote": "<≤16 words, in quotation marks, no attribution>", '
        '"reflection": "<1–2 sentences, ≤40 words, no hashtags>" }\n\n'
        "Rules:\n"
        "- The quote MUST be wrapped in typographic quotes like “...”.\n"
        "- The reflection is concise and Instagram-friendly.\n"
        "- Do not include any extra keys or text besides the JSON."
    )
    raw = gemini_generate(prompt)

    # Parse JSON; if the model wrapped it in prose, extract first {...} block
    try:
        obj = json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start == -1 or end == -1:
            raise RuntimeError(f"Could not parse JSON from Gemini response: {raw[:200]}")
        obj = json.loads(raw[start:end+1])

    verse = (obj.get("quote") or "").strip()
    reflection = (obj.get("reflection") or "").strip()

    # Guardrails: ensure quotes + length caps
    if not verse.startswith(("“", "\"")):
        verse = "“" + verse.strip('“"').strip() + "”"
    if len(verse.split()) > 16:
        verse = " ".join(verse.split()[:16]).rstrip(".!,;:") + "”"

    words = reflection.split()
    if len(words) > 40:
        reflection = " ".join(words[:40]).rstrip(".!,;:") + "."

    return verse, reflection

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
    return base64.b64decode(r.json()["artifacts"][0]["base64"])

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

    y = int(H * 0.16)
    for line in verse_wrapped.split("\n"):
        w, h = draw.textsize(line, font=font_verse)
        draw.text(((W - w) / 2, y), line, font=font_verse, fill=(255, 255, 255))
        y += h + 8

    y += 40
    for line in reflection_wrapped.split("\n"):
        w, h = draw.textsize(line, font=font_reflection)
        draw.text(((W - w) / 2, y), line, font=font_reflection, fill=(235, 235, 235))
        y += h + 6

    return img

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Text via Gemini
    verse, reflection = generate_verse_and_reflection()

    # 2) Image via Stability
    img_bytes = generate_image_bytes(
        f"{THEME}, {STYLE}, instagram composition, detailed, high quality"
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

    repo = os.getenv("GITHUB_REPOSITORY")
    image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"

    payload = {"image_url": image_url, "caption": caption}
    with open("post_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {out_path}\n{caption}")
