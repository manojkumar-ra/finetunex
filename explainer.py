from groq import Groq
import os

# groq for explaining training results
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain_training(summary):
    prompt = f"""Here are the results of fine-tuning a language model:

Base model: {summary['base_model']}
Training examples: {summary['total_examples']}
Epochs: {summary['epochs']}
LoRA rank: {summary['lora_rank']}
Trainable parameters: {summary['trainable_params']:,} out of {summary['total_params']:,} ({summary['trainable_percent']}%)
Final loss: {summary['final_loss']}
Training time: {summary['training_time']} seconds

Explain what these results mean in plain english. Cover:
1. Whether the training went well based on the final loss
2. What the trainable parameter percentage means (LoRA efficiency)
3. Tips for improving results if needed
Keep it under 6 sentences."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "you are an ML engineer. explain fine-tuning results clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"could not generate explanation: {str(e)}"
