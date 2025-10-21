# preprocess/text_normalizer.py
"""
Language-specific normalization with Hindi & English rules.
Extend rules further as needed.
"""
import re
from num2words import num2words

def normalize_en(text):
    """English text normalization"""
    t = text.strip()
    t = t.replace("\u200b", " ")
    # expand simple numbers
    def repl_num(m):
        s = m.group(0)
        try:
            return num2words(int(s))
        except Exception:
            return s
    t = re.sub(r"\b\d+\b", repl_num, t)
    t = re.sub(r"[_\[\]\(\)\{\}<>]", " ", t)
    t = re.sub(r"[^a-zA-Z\s']", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()

def normalize_hi(text):
    """Hindi text normalization (Devanagari cleanup)"""
    t = text.strip()
    t = t.replace("\u200b", " ")
    # remove bracketed annotations
    t = re.sub(r"\[.*?\]|\(.*?\)", "", t)
    # retain only Hindi chars + danda + space
    t = re.sub(r"[^\u0900-\u097F\s।]", "", t)
    # normalize danda and spacing
    t = re.sub(r"।+", "।", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def normalize_by_lang(text, lang="en"):
    if lang.startswith("hi"):
        return normalize_hi(text)
    else:
        return normalize_en(text)
