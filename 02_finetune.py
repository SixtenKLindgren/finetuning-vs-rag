# Finetunar basmodellen med LoRA på Wookieepedia-datan
# Kör på Google Colab med GPU (Runtime -> Change runtime type -> T4 GPU)
# Kräver: pip install transformers datasets accelerate peft bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# GPU check
if not torch.cuda.is_available():
    raise RuntimeError("Ingen GPU hittad. Byt till GPU i Colab: Runtime -> Change runtime type")
print(f"GPU: {torch.cuda.get_device_name(0)}")

MODEL = "Qwen/Qwen2.5-0.5B"
DATA = "wookieepedia_corpus.jsonl"
OUTPUT = "starwars-qwen-lora"

# Steg 1: Ladda basmodellen
print("Laddar basmodellen...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True,
)

# Steg 2: Konfigurera LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Steg 3: Ladda träningsdatan
print("Laddar data...")
dataset = load_dataset("json", data_files=DATA, split="train")
print(f"  {len(dataset)} artiklar")

# Tokenisera texten
def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=1024, padding=False)

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Steg 4: Träna
print("Startar träning...")
training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# Steg 5: Spara
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print(f"Modellen sparad till: {OUTPUT}")
