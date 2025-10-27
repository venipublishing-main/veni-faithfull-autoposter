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
# STABILITY IMAGE GENERATION (resilient + no-text)
# -----------------------------
def generate_image_bytes(prompt, width=1024, height=1024):
    import random

    # Allow override from env; else try a safe fallback list
    primary_engine = os.getenv("STABILITY_ENGINE", "stable-diffusion-v1-6")
    fallback_engines = [
        primary_engine,                        # env-pinned first
        "stable-diffusion-v1-5",               # older but common
        "stable-diffusion-xl-1024-v1-0",       # XL route (still supports JSON base64 with this path)
    ]

    negative = (
        "text, letters, typography, captions, watermark, logo, signature, words, "
        "flat plain background, low detail, low contrast, artifacts"
    )

    headers = {
        "Authorization": f"Bearer {STABILITY_KEY}",
        "Accept": "application/json",
    }

    last_err = None
    for engine in fallback_engines:
        url = f"https://api.stability.ai/v1/generation/{engine}/text-to-image"
        body = {
            "text_prompts": [
                {"text": f"{prompt}, no text, no typography, clean background"},
                {"text": negative, "weight": -1.2}
            ],
            "cfg_scale": 7,
            "width": width,
            "height": height,
            "samples": 1,
            "seed": random.randint(1, 2_147_483_000),
        }
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 404:
                print(f"[Stability] 404 for engine '{engine}' — trying next fallback…")
                continue
            r.raise_for_status()
            data = r.json()
            return base64.b64decode(data["artifacts"][0]["base64"])
        except Exception as e:
            last_err = e
            # Print a short server message (if any) to logs for troubleshooting
            try:
                print(f"[Stability] Engine '{engine}' failed: {r.status_code} {r.text[:200]}")
            except Exception:
                print(f"[Stability] Engine '{engine}' failed: {e}")

    raise RuntimeError(f"All Stability engines failed. Last error: {last_err}")


# -----------------------------
# TEXT OVERLAY (auto-fit + backdrop)
# -----------------------------
import math

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        # Fallback to a common runner font or default
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

def _wrap_to_width(draw, text, font, max_w):
    lines, cur = [], []
    for word in text.split():
        test = (" ".join(cur + [word])).strip()
        w = draw.textlength(test, font=font)
        if w <= max_w or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines

def _auto_fit_block(draw, text, target_w, target_h, max_size, min_size=18, step=2, font_path=FONT_PATH):
    size = max_size
    while size >= min_size:
        font = _load_font(font_path, size)
        lines = _wrap_to_width(draw, text, font, target_w)
        if not lines:
            size -= step
            continue
        # Calculate total height with line spacing ~0.25*size
        line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        total_h = int(len(lines) * (line_h + math.ceil(size * 0.25)))
        max_line_w = max(draw.textlength(l, font=font) for l in lines)
        if total_h <= target_h and max_line_w <= target_w:
            return font, lines, line_h
        size -= step
    # Fallback
    font = _load_font(font_path, min_size)
    lines = _wrap_to_width(draw, text, font, target_w)
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    return font, lines, line_h

def _apply_bottom_gradient(img, strength=220):
    """Subtle bottom gradient (transparent→black) for readability."""
    w, h = img.size
    grad = Image.new("L", (1, h), 0)
    for i in range(h):
        t = max(0, (i - int(h*0.55)) / (h*0.45))  # start ~55% down
        val = int((t**1.8) * strength)
        grad.putpixel((0, i), val)
    alpha = grad.resize((w, h))
    black = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    black.putalpha(alpha)
    return Image.alpha_composite(img.convert("RGBA"), black).convert("RGB")

def draw_centered_text(img, verse, reflection):
    """Replaces the old function with auto-fit, backdrop, stroke, and gradient."""
    img = _apply_bottom_gradient(img)
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Safe margins & layout regions
    margin = int(w * 0.08)
    verse_box = (margin, int(h*0.14), w - margin, int(h*0.59))  # top ~45%
    refl_box  = (margin, int(h*0.62), w - margin, int(h*0.84))  # bottom ~22%

    # Verse (bigger)
    v_draw = ImageDraw.Draw(img, "RGBA")
    v_font, v_lines, v_line_h = _auto_fit_block(
        v_draw, verse,
        target_w=verse_box[2]-verse_box[0],
        target_h=verse_box[3]-verse_box[1],
        max_size=int(w*0.085),  # ~8.5% width as starting point
        font_path=FONT_PATH
    )
    _draw_block_with_backdrop(
        v_draw, (verse_box[0], verse_box[1]),
        v_lines, v_font, v_line_h,
        pad=16, stroke=2, fill=(255,255,255), stroke_fill=(0,0,0), backdrop_alpha=90
    )

    # Reflection (smaller)
    r_draw = ImageDraw.Draw(img, "RGBA")
    r_font, r_lines, r_line_h = _auto_fit_block(
        r_draw, reflection,
        target_w=refl_box[2]-refl_box[0],
        target_h=refl_box[3]-refl_box[1],
        max_size=int(w*0.05),   # ~5% width starting point
        font_path=FONT_PATH
    )
    _draw_block_with_backdrop(
        r_draw, (refl_box[0], refl_box[1]),
        r_lines, r_font, r_line_h,
        pad=14, stroke=2, fill=(235,235,235), stroke_fill=(0,0,0), backdrop_alpha=72
    )
    return img

def _draw_block_with_backdrop(draw, xy, lines, font, line_h,
                              stroke=2, fill=(255,255,255), stroke_fill=(0,0,0),
                              pad=16, backdrop_alpha=90):
    """Translucent rectangle behind text + stroked text for high contrast."""
    x, y = xy
    widths = [draw.textlength(l, font=font) for l in lines] if lines else [0]
    block_w = int(max(widths)) if widths else 0
    block_h = int(len(lines) * (line_h + math.ceil(font.size * 0.25)))

    if block_w > 0 and block_h > 0 and backdrop_alpha > 0:
        rect = [x - pad, y - pad, x + block_w + pad, y + block_h + pad]
        draw.rectangle(rect, fill=(0, 0, 0, backdrop_alpha))

    cy = y
    for l in lines:
        draw.text((x, cy), l, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        cy += line_h + math.ceil(font.size * 0.25)

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
