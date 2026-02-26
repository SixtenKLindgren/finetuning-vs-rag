# RAG-system som söker i Wookieepedia och svarar på frågor
# Kräver: pip install rank_bm25 torch transformers

import json
import re
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS = "wookieepedia_corpus.jsonl"
MODEL = "Qwen/Qwen2.5-0.5B"


def tokenize(text):
    return re.findall(r'\w+', text.lower())


# Steg 1: Ladda alla artiklar 
print("Laddar data...")
articles = []
with open(CORPUS, 'r', encoding='utf-8') as f:
    for rad in f:
        articles.append(json.loads(rad))
print(f"  {len(articles):,} artiklar laddade")

# Steg 2: Bygg sökindex med BM25
print("Bygger sökindex...")
tokeniserad_korpus = [tokenize(a['text']) for a in articles]
bm25 = BM25Okapi(tokeniserad_korpus)
print("Index klart")

# Steg 3: Ladda språkmodellen
print("Laddar modellen...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model = model.to(device)
print(f"  Modell laddad på {device}")


def search(question, top_k=3):
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [articles[i] for i in best]


def answer(question):

    # Hämta de 3 mest relevanta artiklarna
    hittade = search(question, top_k=3)
    print(f"  Hämtade: {', '.join(a['title'] for a in hittade)}")

    # Bygg prompt med artiklarna som kontext
    context = ""
    for i, a in enumerate(hittade, 1):
        context += f"[Källa {i}: {a['title']}]\n{a['text'][:2000]}\n\n"

    prompt = f"""Use the following information to answer the question.

{context}
Question: {question}
Answer:"""

    # Generera svar
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()

