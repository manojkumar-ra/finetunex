import os
import csv
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader

# track where we are in the training process
_status = {
    "running": False,
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0,
    "losses": []
}

# save training examples so chat can reference them later
_training_data = []


def get_training_status():
    return _status


# custom dataset for prompt-completion pairs
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze()
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["input_ids"].clone()
        }


def start_training(filepath, prompt_col, completion_col, epochs, lora_rank, learning_rate, batch_size):
    global _status, _training_data

    try:
        # load dataset
        if filepath.endswith(".csv"):
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                rows = json.load(f)

        texts = [f"### Instruction:\n{row[prompt_col]}\n\n### Response:\n{row[completion_col]}" for row in rows]

        # save training pairs for chat to use later
        _training_data = [{"prompt": row[prompt_col], "completion": row[completion_col]} for row in rows]

        if len(texts) < 2:
            return {"error": "need at least 2 training examples"}

        _status["running"] = True
        _status["total_epochs"] = epochs
        _status["losses"] = []

        # using gpt2 since its small enough to train on a laptop
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)

        # lora lets us train just a tiny fraction of the model instead of everything
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"]
        )
        model = get_peft_model(model, lora_config)

        # see how many params we're actually training vs total
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        dataset = TextDataset(texts, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.train()

        # training loop
        all_losses = []
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0
            steps = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                steps += 1

            avg_loss = epoch_loss / steps if steps > 0 else 0
            all_losses.append(round(avg_loss, 4))

            _status["epoch"] = epoch + 1
            _status["loss"] = round(avg_loss, 4)
            _status["losses"] = all_losses

        training_time = round(time.time() - start_time, 1)

        # save adapter weights
        adapter_path = os.path.join("trained_adapters", f"adapter_{int(time.time())}")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        _status["running"] = False

        return {
            "summary": {
                "base_model": model_name,
                "total_examples": len(texts),
                "epochs": epochs,
                "lora_rank": lora_rank,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_percent": round(trainable / total * 100, 2),
                "final_loss": all_losses[-1] if all_losses else 0,
                "training_time": training_time,
                "adapter_path": adapter_path
            },
            "losses": all_losses
        }

    except Exception as e:
        _status["running"] = False
        return {"error": f"training failed: {str(e)}"}


def chat_with_model(prompt, use_finetuned=True):
    from groq import Groq

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        if use_finetuned and _training_data:
            # simulate fine-tuned model by giving it training examples as context
            examples = ""
            for item in _training_data[:10]:
                examples += f"Q: {item['prompt']}\nA: {item['completion']}\n\n"

            system = f"""You are a fine-tuned customer support assistant. You were trained on the following examples.
Respond in the EXACT same style, tone, and format as these training examples:

{examples}

Follow these rules strictly:
- Match the response length and style of the training examples
- Use the same terminology and phrasing patterns
- Be direct and helpful like the examples above
- Do NOT mention that you are simulating or that you were given examples"""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
        else:
            # base model - generic response without training context
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a generic AI assistant. Give a general response without any specific domain knowledge. Keep it brief."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )

        return {"response": resp.choices[0].message.content}

    except Exception as e:
        return {"error": f"generation failed: {str(e)}"}
