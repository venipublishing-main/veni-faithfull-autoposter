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

DEEPAI_KEY     = os.getenv("DEEPAI_KEY")
STABILITY_KEY  = os.getenv("STABILITY_API_KEY")

if not DEEPAI_KEY:
    raise RuntimeError("Missing DEEPAI_KEY (DeepAI API key)")
if not STABILITY_KEY:
    raise RuntimeError("Missing STABILITY_API_KEY (Stability AI key)")

# -----------------------------
# DEEPAI TEXT GENERATION
# -----------------------------
def deepai_generate(prompt: str) -> str:
    url = "https://api.deepai.org/api/text-generator"
    headers = {"api-key": DEEPAI_KEY}
    r = requests.post(url, data={"text": prompt}, headers=headers, timeout=45)
    r.raise_for_status()
    data = r.json()
    out = (data.get("output") or "").strip()
    if not out:
        raise RuntimeError(f"DeepAI returned empty output for prompt: {prompt[:80]}...")
    return out

def generate_verse_and_reflection():
    # Quote: short, generic (no attribution), fits on image
    quote_prompt = (
        f"Write one short, powerful quote about {THEME}. "
        f"It must be generic (no attribution), wrapped in quotation marks, and 16 words or fewer."
    )
    verse = deepai_generate(quote_prompt)
    # enforce wrapping quotes if model forgot
    if not verse.startswith(("“", "\"")):
        verse = "“" + verse.strip('“"').strip() + "”"
    # Trim overly long outputs (safety)
    if len(verse.split()) > 16:
        verse = " ".join(verse.split()[:16])
        if not verse.endswith("”"):
            verse = verse.rstrip(".!,;:") + "”"

    # Reflection: 1–2 sentences, short, IG-friendly
    reflection_prompt = (
        f"Write a concise 1–2 sentence reflection that expands on this quote for Instagram. "
        f"Tone: {TONE}. <= 40 words. Avoid hashtags.\nQuote: {verse}"
    )
    reflection = deepai_generate(reflection_prompt)
    # keep it tight
    words = reflection.split()
    if len(words) > 40:
        reflection = " ".join(words[:40]).rstrip(".!,;:") + "."

    return verse.strip(), reflection.strip()

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

    # 1) Text via DeepAI
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
