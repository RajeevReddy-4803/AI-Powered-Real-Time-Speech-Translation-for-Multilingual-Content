"""
Lightweight, memory-efficient text normalization for Hindi and English.
Designed for ASR preprocessing (Whisper, IndicWav2Vec).
"""

import re
from num2words import num2words

__all__ = ["normalize_hi", "normalize_en", "normalize_by_lang"]


# =============================
# 🔤 English Normalization
# =============================
def normalize_en(text: str) -> str:
    """
    Normalize English text:
    - Removes punctuation/symbols
    - Converts digits to words
    - Keeps only a-z, spaces, and apostrophes
    - Lowercases output
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    t = text.strip()
    t = t.replace("\u200b", " ")  # remove zero-width space

    # Replace numbers with words (skip if conversion fails)
    def repl_num(m):
        s = m.group(0)
        try:
            return num2words(int(s))
        except Exception:
            return s

    t = re.sub(r"\b\d+\b", repl_num, t)

    # Remove symbols and normalize spacing
    t = re.sub(r"[_\[\]\(\)\{\}<>\"“”‘’]", " ", t)
    t = re.sub(r"[^a-zA-Z\s']", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()


# =============================
# 🇮🇳 Hindi Normalization
# =============================
def normalize_hi(text: str) -> str:
    """
    Normalize Hindi text:
    - Removes non-Devanagari characters
    - Keeps danda (।) as sentence separator
    - Collapses extra spaces
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    t = text.strip()
    t = t.replace("\u200b", " ")

    # Remove bracketed or parenthetical content
    t = re.sub(r"\[.*?\]|\(.*?\)", "", t)

    # Keep only Devanagari + danda + space
    t = re.sub(r"[^\u0900-\u097F\s।]", "", t)
    t = re.sub(r"।+", "।", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# =============================
# 🌐 Auto Router
# =============================
def normalize_by_lang(text: str, lang: str = "en") -> str:
    """
    Normalize text based on language.
    Supports 'hi' (Hindi) and 'en' (English).
    """
    if lang.startswith("hi"):
        return normalize_hi(text)
    elif lang.startswith("en"):
        return normalize_en(text)
    else:
        # fallback — choose by script detection
        if re.search(r"[\u0900-\u097F]", text):
            return normalize_hi(text)
        return normalize_en(text)


# =============================
# 🧪 Quick Test
# =============================
if __name__ == "__main__":
    samples = {
        "en": "There were 3 birds sitting on a 100-year-old tree!",
        "hi": "रॉकेट लांचर से संसद भवन पर मिसाइलें दागी गईं ।"
    }

    for lang, txt in samples.items():
        print(f"{lang.upper()} → {normalize_by_lang(txt, lang)}")
