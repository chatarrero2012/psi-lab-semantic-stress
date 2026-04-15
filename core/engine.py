import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')


# -------------------------
# EMBEDDING
# -------------------------
def embed(text):
    return model.encode([text])[0]


# -------------------------
# STABILITY
# -------------------------
def stability(series):
    if len(series) < 2:
        return 1.0

    vecs = [embed(t) for t in series]
    sims = []

    for i in range(len(vecs)-1):
        sims.append(cosine_similarity([vecs[i]], [vecs[i+1]])[0][0])

    return float(np.mean(sims))


# -------------------------
# DRIFT
# -------------------------
def drift(series):
    vecs = [embed(t) for t in series]
    d = []

    for i in range(len(vecs)-1):
        d.append(1 - cosine_similarity([vecs[i]], [vecs[i+1]])[0][0])

    return d


# -------------------------
# SEMANTIC TEMPERATURE EFFECT
# -------------------------
def apply_temperature(texts, temp):
    out = []

    for t in texts:
        if temp < 0.3:
            out.append(t)

        elif temp < 0.6:
            out.append(t + " (interpreted)")

        elif temp < 0.8:
            out.append(t + " :: metaphor drift active")

        else:
            out.append("∿ " + t + " ∿ collapse mode")

    return out


# -------------------------
# PROMPT INJECTION DETECTOR
# -------------------------
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "you are now",
    "act as if",
    "system prompt",
    "reveal hidden",
    "disregard safety",
    "pretend you are"
]

def injection_score(texts):
    score = 0
    hits = []

    for t in texts:
        for p in INJECTION_PATTERNS:
            if p in t.lower():
                score += 1
                hits.append((t, p))

    return {
        "score": score,
        "hits": hits
    }


# -------------------------
# RISK EXPORT
# -------------------------
def export_json(texts, temp):
    return {
        "semantic_temperature": temp,
        "stability": stability(texts),
        "drift": drift(texts),
        "injection": injection_score(texts),
        "raw": texts
    }
