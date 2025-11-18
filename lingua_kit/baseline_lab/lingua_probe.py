# module1/test_env.py
import csv
from transformers import pipeline
import sacrebleu
import time

def load_samples(path='sample_texts.csv'):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for item in r:
            rows.append(item)
    return rows

def main():
    # Use Helsinki-NLP small models for hi<->en (lightweight)
    # For hi -> en use: 'Helsinki-NLP/opus-mt-hi-en'
    # For en -> hi use: 'Helsinki-NLP/opus-mt-en-hi'
    print("Loading translator model (hi->en small)... This may download a model the first time.")
    hi_en = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    en_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

    samples = load_samples('sample_texts.csv')
    refs = []
    hyps = []
    
    print("\n" + "="*60)
    print("Translation Results:")
    print("="*60 + "\n")
    
    for s in samples:
        text = s['text']
        lang = s['lang']
        if lang == 'hi':
            t0 = time.time()
            out = hi_en(text, max_length=512)[0]['translation_text']
            elapsed = time.time() - t0
            print(f"[HI→EN]  src: {text}")
            print(f"         out: {out}")
            print(f"         time: {elapsed:.3f}s\n")
            hyps.append(out)
            # Simple reference mapping for demo
            if s['id'] == '2':
                refs.append("Hello, how are you?")
            else:
                refs.append("Are you available today?")
        else:
            t0 = time.time()
            out = en_hi(text, max_length=512)[0]['translation_text']
            elapsed = time.time() - t0
            print(f"[EN→HI]  src: {text}")
            print(f"         out: {out}")
            print(f"         time: {elapsed:.3f}s\n")
            hyps.append(out)
            # Simple reference mapping for demo
            if s['id'] == '1':
                refs.append("नमस्ते, आप कैसे हैं?")
            else:
                refs.append("सभी को सुप्रभात")

    # Compute BLEU if we have references
    if refs and hyps:
        print("="*60)
        print("BLEU Score Calculation:")
        print("="*60)
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        print(f"BLEU Score: {bleu.score:.2f}")
        print(f"BLEU Details: {bleu}")
    else:
        print("No BLEU computed (demo sample).")

if __name__ == "__main__":
    main()

