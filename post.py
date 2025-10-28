#!/usr/bin/env python3
import os, json, base64, textwrap, math, random, time, io
from datetime import datetime, timedelta
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO

# Optional CV libs (required for AI-aware placement)
import numpy as np
import cv2

# ======================================================
# CONFIGURATION
# ======================================================
OUT_DIR = "out"
ASSETS_DIR = "assets"
STATE_DIR = "state"
LIBRARY_PATH = os.getenv("LIBRARY_PATH", f"{ASSETS_DIR}/veni_library.json")
HISTORY_PATH = f"{STATE_DIR}/history.json"

FONT_PATH = "assets/fonts/IMFellEnglishSC-Regular.ttf"
OVERLAY_TEXT_ON_IMAGE = True  # toggle to post clean image if False

# Env defaults
THEME = os.getenv("THEME", "discipline, focus, perseverance").lower()
TONE  = os.getenv("TONE",  "motivational, concise, modern")
ATTRIBUTION_MODE = os.getenv("ATTRIBUTION_MODE", "hybrid").lower()  # hybrid | ai
BIBLE_TRANSLATION = os.getenv("BIBLE_TRANSLATION", "auto").upper()  # KJV | WEB | AUTO
NO_REPEAT_DAYS = int(os.getenv("NO_REPEAT_DAYS", "21"))

GEMINI_KEY        = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "400"))
STABILITY_KEY     = os.getenv("STABILITY_API_KEY", "").strip()

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")
if not STABILITY_KEY:
    raise RuntimeError("Missing STABILITY_API_KEY")

# ======================================================
# STYLE FAMILIES (theme → visual aesthetic)
# ======================================================
STYLE_FAMILIES = {
    "ascetic": (
        "monastic minimalism, aged parchment texture, muted warm light, "
        "meditative atmosphere, candlelight shadows, classical composition"
    ),
    "warrior": (
        "cinematic chiaroscuro, metal and stone texture, dramatic contrast, "
        "embers and dust, determination, dynamic lighting"
    ),
    "contemplative": (
        "soft focus, natural light through fog, tranquil balance, "
        "subtle color harmony, zen-like simplicity"
    ),
    "visionary": (
        "ethereal glow, sunrise tones, clarity and optimism, "
        "spacious composition, bright volumetric light, hopeful atmosphere"
    ),
    "mystical": (
        "deep indigo palette, star-lit ambience, sacred geometry hints, "
        "symbolic surrealism, dreamlike textures"
    ),
}

def pick_style_from_theme(theme: str) -> str:
    if "discipline" in theme or "ascetic" in theme:
        return STYLE_FAMILIES["ascetic"]
    if "focus" in theme or "perseverance" in theme or "warrior" in theme:
        return STYLE_FAMILIES["warrior"]
    if "reflection" in theme or "mind" in theme or "thought" in theme or "peace" in theme:
        return STYLE_FAMILIES["contemplative"]
    if "vision" in theme or "light" in theme or "clarity" in theme or "hope" in theme:
        return STYLE_FAMILIES["visionary"]
    return STYLE_FAMILIES["mystical"]

# ======================================================
# LIBRARY + HISTORY (Hybrid mode)
# ======================================================
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def theme_tags(theme_str: str):
    return [t.strip().lower() for t in theme_str.split(",") if t.strip()]

def filter_entries_by_tags(entries, tags):
    if not tags:
        return entries[:]
    out = []
    tagset = set(tags)
    for e in entries:
        etags = set((e.get("tags") or []))
        if etags & tagset:
            out.append(e)
    return out or entries[:]  # fallback if none matched

def is_recent(id_, history, days):
    now = datetime.utcnow()
    for item in history:
        if item.get("id") == id_:
            try:
                dt = datetime.fromisoformat(item.get("ts"))
                if now - dt < timedelta(days=days):
                    return True
            except Exception:
                continue
    return False

def select_entry(library, theme, history, days, prefer_overlay=True):
    tags = theme_tags(theme)
    pool = filter_entries_by_tags(library, tags)
    # optional: prefer allowed_overlay for image text
    if prefer_overlay:
        overlay_pool = [e for e in pool if e.get("allowed_overlay", True)]
        if overlay_pool:
            pool = overlay_pool
    # filter out recent ids
    pool = [e for e in pool if not is_recent(e.get("id"), history, days)]
    if not pool:
        pool = library[:]  # fallback if all are recent
    return random.choice(pool)

# ======================================================
# GEMINI REFLECTION
# ======================================================
def gemini_reflection(text_for_context: str, ref_or_author: str, kind: str) -> str:
    """
    1–3 sentences, ≤ ~70 words total, expanding on the meaning.
    Avoids quoting or paraphrasing the given text.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    sys = (
        "You write contemplative Instagram reflections. "
        "Output 1 to 3 sentences total (no more), at most about 70 words. "
        "Do NOT restate or quote the provided text. Avoid hashtags and emojis."
    )
    usr = (
        f"Kind: {kind}\n"
        f"Source: {ref_or_author}\n"
        f"Text:\n{text_for_context}\n\n"
        f"Write a concise reflection for modern readers. Keep it warm, clear, and practical."
    )
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": sys}]},
        "contents": [{"role": "user", "parts": [{"text": usr}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": GEMINI_MAX_TOKENS}
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    txt = (
        (data.get("candidates") or [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    ).strip()

    # Hard cap ~70 words (safety net)
    words = txt.split()
    if len(words) > 70:
        txt = " ".join(words[:70]).rstrip(".!,;:") + "."
    return txt or "Let this truth shape your next step today."
    
def gemini_closer(text_for_context: str, kind: str) -> str:
    """
    One short closing line (<= 12 words) that invites contemplation or action.
    No hashtags/emojis. Returns empty string on failure.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    sys = (
        "You write a single short closing line for an Instagram caption. "
        "At most 12 words. No hashtags or emojis."
    )
    usr = (
        f"Kind: {kind}\n"
        f"Text:\n{text_for_context}\n\n"
        f"Write ONE short closing line that invites contemplation or gentle action."
    )
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": sys}]},
        "contents": [{"role": "user", "parts": [{"text": usr}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 40}
    }
    try:
        r = requests.post(url, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        line = (
            (data.get("candidates") or [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        ).strip()
        # Trim to 12 words just in case
        words = line.split()
        if len(words) > 12:
            line = " ".join(words[:12]).rstrip(".!,;:") + "."
        # Guardrail: avoid empty or repeated punctuation
        if len(line) < 3:
            return ""
        return line
    except Exception:
        return ""

# ======================================================
# STABILITY IMAGE GENERATION (resilient + no-text)
# ======================================================
def generate_image_bytes(prompt, width=1024, height=1024):
    primary_engine = os.getenv("STABILITY_ENGINE", "stable-diffusion-v1-6")
    fallback_engines = [primary_engine, "stable-diffusion-v1-5", "stable-diffusion-xl-1024-v1-0"]
    negative = (
        "text, letters, typography, captions, watermark, logo, signature, words, "
        "flat plain background, low detail, low contrast, artifacts"
    )
    headers = {"Authorization": f"Bearer {STABILITY_KEY}", "Accept": "application/json"}
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
            try:
                print(f"[Stability] Engine '{engine}' failed: {r.status_code} {r.text[:200]}")
            except Exception:
                print(f"[Stability] Engine '{engine}' failed: {e}")
    raise RuntimeError(f"All Stability engines failed. Last error: {last_err}")

# ======================================================
# AI-AWARE LAYOUT (saliency + brightness + golden ratio)
# ======================================================
def _saliency_like(gray_u8: np.ndarray) -> np.ndarray:
    lap = cv2.Laplian(gray_u8, cv2.CV_32F, ksize=3)  # typo intentional? fix to Laplacian
    # Correct spelling:
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    gx  = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    sob = cv2.magnitude(gx, gy)
    sal = 0.6 * np.abs(lap) + 0.4 * sob
    sal = cv2.GaussianBlur(sal, (0,0), 3)
    sal -= sal.min()
    if sal.max() > 1e-6:
        sal /= sal.max()
    return sal

def _brightness(gray_u8: np.ndarray) -> np.ndarray:
    return gray_u8.astype(np.float32) / 255.0

def _score_region(sal_map: np.ndarray, bright_map: np.ndarray, rect):
    x1, y1, x2, y2 = rect
    roi_sal = sal_map[y1:y2, x1:x2]
    roi_br  = bright_map[y1:y2, x1:x2]
    if roi_sal.size == 0:
        return 1e9
    mean_sal = float(roi_sal.mean())
    std_sal  = float(roi_sal.std())
    mean_br  = float(roi_br.mean())
    penalty = 0.0
    if mean_br < 0.25:
        penalty += (0.25 - mean_br) * 0.6
    if mean_br > 0.90:
        penalty += (mean_br - 0.90) * 0.6
    return 0.7 * mean_sal + 0.3 * std_sal + penalty

def _find_best_text_box(pil_img, box_rel=(0.78, 0.36), margin_rel=0.08):
    w, h = pil_img.size
    img_rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sal = _saliency_like(gray)
    br  = _brightness(gray)

    margin = int(w * margin_rel)
    bw = max(int(w * box_rel[0]), 100)
    bh = max(int(h * box_rel[1]), 100)

    xs = [margin, (w - bw)//2, max(w - bw - margin, margin)]
    ys = [margin, (h - bh)//2, max(h - bh - margin, margin)]
    candidates = [(xx, yy, xx + bw, yy + bh) for yy in ys for xx in xs]

    # Golden ratio candidates
    gx = int((w - bw) * 0.382); gy = int((h - bh) * 0.382)
    candidates += [(gx, gy, gx + bw, gy + bh),
                   (w - bw - gx, h - bh - gy, w - gx, h - gy)]

    best_rect, best_score = None, 1e9
    for rect in candidates:
        score = _score_region(sal, br, rect)
        if score < best_score:
            best_score, best_rect = score, rect

    # Clamp
    x1, y1, x2, y2 = best_rect
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w)); y2 = max(y1+1, min(y2, h))
    return (x1, y1, x2, y2)

# ======================================================
# TYPOGRAPHY & RENDER
# ======================================================
def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
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
            lines.append(" ".join(cur)); cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines

def _auto_fit_block(draw, text, target_w, target_h, max_size, min_size=18, step=2, font_path=FONT_PATH):
    size = max_size
    while size >= min_size:
        font = _load_font(font_path, size)
        lines = _wrap_to_width(draw, text, font, target_w)
        if not lines:
            size -= step; continue
        line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        total_h = int(len(lines) * (line_h + math.ceil(size * 0.25)))
        max_line_w = max(draw.textlength(l, font=font) for l in lines)
        if total_h <= target_h and max_line_w <= target_w:
            return font, lines, line_h
        size -= step
    font = _load_font(font_path, min_size)
    lines = _wrap_to_width(draw, text, font, target_w)
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    return font, lines, line_h

def _apply_bottom_gradient(img, strength=220):
    w, h = img.size
    grad = Image.new("L", (1, h), 0)
    for i in range(h):
        t = max(0, (i - int(h*0.55)) / (h*0.45))
        val = int((t**1.8) * strength)
        grad.putpixel((0, i), val)
    alpha = grad.resize((w, h))
    black = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    black.putalpha(alpha)
    return Image.alpha_composite(img.convert("RGBA"), black).convert("RGB")

def _pick_text_colors(img, rect):
    x1, y1, x2, y2 = rect
    region = img.crop((x1, y1, x2, y2)).resize((1,1), Image.BOX)
    r, g, b = region.getpixel((0,0))
    # luma
    luma = 0.2126*r + 0.7152*g + 0.0722*b
    if luma < 110:
        fill = (245, 245, 245)   # light text
        shadow = (0, 0, 0, 120)
    else:
        fill = (25, 25, 25)
        shadow = (0, 0, 0, 80)
    return fill, shadow

def _rounded_backdrop(mask_size, rect, radius=40, alpha=84):
    w, h = mask_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = rect
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=alpha)
    mask = mask.filter(ImageFilter.GaussianBlur(2.0))
    return mask

def draw_quote_and_attrib(img, quote_text, attribution):
    img = _apply_bottom_gradient(img)
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    verse_box = _find_best_text_box(img, box_rel=(0.78, 0.36), margin_rel=0.08)
    fill, shadow = _pick_text_colors(img, verse_box)

    # Typography hierarchy
    q_max = int(w * 0.10)
    a_ratio = 0.75  # attribution smaller
    q_font, q_lines, q_line_h = _auto_fit_block(
        draw, quote_text, target_w=verse_box[2]-verse_box[0], target_h=int((verse_box[3]-verse_box[1]) * 0.78),
        max_size=q_max, font_path=FONT_PATH
    )
    # Attribution gets remaining space beneath quote
    attrib_max = max(18, int(q_font.size * a_ratio))
    a_font, a_lines, a_line_h = _auto_fit_block(
        draw, attribution, target_w=verse_box[2]-verse_box[0], target_h=int((verse_box[3]-verse_box[1]) * 0.22),
        max_size=attrib_max, font_path=FONT_PATH
    )

    # Backdrop under the combined block
    q_block_h = int(len(q_lines) * (q_line_h + math.ceil(q_font.size * 0.25)))
    a_block_h = int(len(a_lines) * (a_line_h + math.ceil(a_font.size * 0.25)))
    pad = 20
    rect = (verse_box[0], verse_box[1], verse_box[0] + int(max(
        max(draw.textlength(l, font=q_font) for l in q_lines) if q_lines else 0,
        max(draw.textlength(l, font=a_font) for l in a_lines) if a_lines else 0
    )) + pad*2, verse_box[1] + q_block_h + a_block_h + pad*3)

    mask = _rounded_backdrop((w, h), rect, radius=40, alpha=84)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    overlay.putalpha(mask)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Draw text with soft shadow
    x = rect[0] + pad
    y = rect[1] + pad
    for line in q_lines:
        draw.text((x+2, y+2), line, font=q_font, fill=shadow)
        draw.text((x, y), line, font=q_font, fill=fill)
        y += q_line_h + math.ceil(q_font.size * 0.25)
    y += pad // 2
    # attribution slightly letter-spaced (simple trick: add thin spaces)
    attrib_vis = attribution.replace(" ", "  ")
    for line in _wrap_to_width(draw, attrib_vis, a_font, rect[2]-rect[0]-pad*2):
        draw.text((x+1, y+1), line, font=a_font, fill=(0,0,0,90))
        draw.text((x, y), line, font=a_font, fill=tuple(int(c*0.9) for c in fill[:3]))
        y += a_line_h + math.ceil(a_font.size * 0.25)

    return img

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    # 0) Load library + history
    library = load_json(LIBRARY_PATH, default=[])
    history = load_json(HISTORY_PATH, default=[])

    # 1) Select entry (Hybrid) or fallback to AI text if library missing
    if ATTRIBUTION_MODE == "hybrid" and library:
        entry = select_entry(library, THEME, history, NO_REPEAT_DAYS, prefer_overlay=True)
        kind = entry.get("kind", "bible")

        if kind == "bible":
            ref = entry.get("ref", "")
            trans = entry.get("translations", {})
            # choose translation
            t_pref = BIBLE_TRANSLATION
            text = None
            if t_pref == "KJV":
                text = trans.get("KJV")
            elif t_pref == "WEB":
                text = trans.get("WEB")
            else:  # AUTO
                text = trans.get("WEB") if random.random() < 0.5 and trans.get("WEB") else trans.get("KJV")
            if not text:
                text = trans.get("KJV") or trans.get("WEB") or ""
            quote_text = text.strip()
            attribution = f"— {ref} ({'WEB' if text == trans.get('WEB') and trans.get('WEB') else 'KJV'})"

        elif kind == "historical":
            quote_text = (entry.get("text") or "").strip()
            attribution = f"— {entry.get('author','Unknown')}".strip()

        else:  # original
            quote_text = (entry.get("text") or "").strip()
            attribution = "— Inspired Reflection"

        # guardrail: if overlay too long, push quote to caption only (still generate image)
        words = quote_text.split()
        overlay_ok = entry.get("allowed_overlay", True) and len(words) <= 32

        # 2) Reflection via Gemini
        reflection = gemini_reflection(quote_text, attribution.lstrip("— ").strip(), kind)

        # 3) Style (theme-mapped, but allow verse style_hint to override)
        style = pick_style_from_theme(THEME)
        if entry.get("style_hint"):
            style = STYLE_FAMILIES.get(entry["style_hint"], style)

        # 4) Image via Stability (context-aware prompt)
        hint = " ".join(quote_text.split()[:6])
        img_prompt = (
            f"Concept art inspired by themes of {THEME}; "
            f"visualize the feeling of '{hint}' without any written text. "
            f"Visual style: {style}. High-contrast, richly textured, "
            f"cinematic depth of field, professional Instagram composition."
        )
        img_bytes = generate_image_bytes(img_prompt)
        base_img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # 5) Overlay (quote + attribution) or clean image
        if OVERLAY_TEXT_ON_IMAGE and overlay_ok:
            final_img = draw_quote_and_attrib(base_img, f"“{quote_text.strip('“”"')}”", attribution)
        else:
            final_img = base_img

        # 6) Save output
        today = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"{today}.jpg"
        out_path = os.path.join(OUT_DIR, filename)
        final_img.save(out_path, "JPEG", quality=95)

        # 7) Caption
        hashtags = "#Discipline #Focus #Perseverance #DailyMotivation"

        # Generate a short closing line (safe to fail silently)
            closer = gemini_closer(quote_text, kind)
            closer_block = f"\n{closer}" if closer else ""

        caption = (
                    f"“{quote_text.strip('“”\"')}”\n"
                    f"{attribution}\n\n"
                    f"{reflection}"
                    f"{closer_block}\n\n"
                    f"{hashtags}"
                )

        # 8) Payload
        repo = os.getenv("GITHUB_REPOSITORY", "") or "owner/repo"
        image_url = f"https://raw.githubusercontent.com/{repo}/main/{out_path.replace(os.sep, '/')}"
        payload = {"image_url": image_url, "caption": caption}
        with open("post_payload.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 9) Update history
        history.append({"id": entry.get("id"), "ts": datetime.utcnow().isoformat()})
        # prune history older than 60 days
        cutoff = datetime.utcnow() - timedelta(days=60)
        history = [h for h in history if datetime.fromisoformat(h["ts"]) >= cutoff]
        save_json(HISTORY_PATH, history)

        print(f"✅ Generated {out_path}\n{caption}")

    else:
        # Fallback to AI-only quote + reflection (previous behavior)
        raise RuntimeError(
            "Hybrid mode requested but library not found or empty. "
            "Ensure assets/veni_library.json exists with entries."
        )
