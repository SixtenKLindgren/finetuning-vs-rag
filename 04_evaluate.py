# Kör quizet på alla tre system och jämför resultaten
# Kräver: pip install torch transformers peft rank_bm25

import json
import re
import torch
from datetime import datetime
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL = "Qwen/Qwen2.5-0.5B"
LORA = "starwars-qwen-lora"
CORPUS = "wookieepedia_corpus.jsonl"
QUESTIONS = "test_questions.json"

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def check_answer(correct, model_answer):
    """Checks if the correct answer is present in the model's answer."""
    correct = correct.lower().strip()
    correct = re.sub(r'[^\w\s]', '', correct)
    model_answer = re.sub(r'[^\w\s]', '', model_answer.lower())
    return correct in model_answer

def generate(model, tokenizer, prompt, device):
    """Generates an answer from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()



def question_to_prompt(question):
    q = question.rstrip('?')
    if q.startswith("What is "):
        return q.replace("What is ", "") + " is"
    elif q.startswith("Who is "):
        return q.replace("Who is ", "") + " is"
    elif q.startswith("Which "):
        return q.replace("Which ", "The ") + " is"
    elif q.startswith("What "):
        return q.replace("What ", "The ") + " is"
    else:
        return q + " -"

print("Loading questions...")
with open(QUESTIONS, 'r', encoding='utf-8') as f:
    questions = json.load(f)['questions']
print(f"  {len(questions)} questions")

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
base_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)


print("Loading finetuned model...")
finetuned_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
finetuned_model = PeftModel.from_pretrained(finetuned_model, LORA)

 # Select GPU/CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
base_model = base_model.to(device)
finetuned_model = finetuned_model.to(device)
print(f"  Models on {device}")


print("Building RAG index...")
articles = []
with open(CORPUS, 'r', encoding='utf-8') as f:
    for line in f:
        articles.append(json.loads(line))
tokenized = [tokenize(a['text']) for a in articles]
bm25 = BM25Okapi(tokenized)
print(f"  {len(articles):,} articles indexed")



def rag_answer(question):
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    context = ""
    for i, idx in enumerate(best, 1):
        context += f"[Source {i}: {articles[idx]['title']}]\n{articles[idx]['text'][:2000]}\n\n"

    prompt = f"Use the following information to answer the question.\n\n{context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs, max_new_tokens=150, do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()



# Kör quizet
print("\nRunning evaluation...")
results = []

for i, q in enumerate(questions):
    print(f"\r  [{i+1}/100] {q['question'][:50]}...", end="", flush=True)

    prompt = question_to_prompt(q['question'])

    base_answer = generate(base_model, tokenizer, prompt, device)
    finetuned_answer = generate(finetuned_model, tokenizer, prompt, device)
    rag_ans = rag_answer(q['question'])

    results.append({
        'id': q['id'],
        'category': q['category'],
        'question': q['question'],
        'correct_answer': q['answer'],
        'base': base_answer[:500],
        'finetuned': finetuned_answer[:500],
        'rag': rag_ans[:500],
        'base_correct': check_answer(q['answer'], base_answer),
        'finetuned_correct': check_answer(q['answer'], finetuned_answer),
        'rag_correct': check_answer(q['answer'], rag_ans),
    })


# Resultat

print("\n")

base_score = sum(1 for r in results if r['base_correct'])
finetuned_score = sum(1 for r in results if r['finetuned_correct'])
rag_score = sum(1 for r in results if r['rag_correct'])

print("=" * 50)
print("RESULTS")
print("=" * 50)
print(f"  Base model:   {base_score}/100 ({base_score}%)")
print(f"  Finetuned:    {finetuned_score}/100 ({finetuned_score}%)")
print(f"  RAG:          {rag_score}/100 ({rag_score}%)")


print(f"\n{'Category':<22} {'Base':>5} {'FT':>5} {'RAG':>5}")
print("-" * 40)
for cat in sorted(set(r['category'] for r in results)):
    cat_results = [r for r in results if r['category'] == cat]
    b = sum(1 for r in cat_results if r['base_correct'])
    f = sum(1 for r in cat_results if r['finetuned_correct'])
    rr = sum(1 for r in cat_results if r['rag_correct'])
    print(f"  {cat:<20} {b}/10  {f}/10  {rr}/10")


output = f"results_{datetime.now().strftime('%Y%m%d')}.json"
with open(output, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nDetails saved to: {output}")
