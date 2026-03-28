![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![LoRA](https://img.shields.io/badge/LoRA-PEFT-FF6F00?style=for-the-badge)

# FineTuneX

LLM fine-tuning platform that lets you upload custom training data, fine-tune GPT-2 using LoRA adapters, and test your model through a chat interface — all from the browser.

## Features

- Upload CSV or JSON training datasets with prompt/completion pairs
- Fine-tune GPT-2 using LoRA (Parameter-Efficient Fine-Tuning)
- Configure training parameters — epochs, LoRA rank, learning rate, batch size
- Live training loss curve visualization
- AI-powered explanation of training results
- Chat interface to test fine-tuned model vs base model side by side
- Training history saved to MySQL

## Tech Stack

- **Model**: GPT-2 (124M parameters)
- **Fine-tuning**: LoRA via PEFT library
- **Training**: PyTorch, HuggingFace Transformers
- **Backend**: FastAPI
- **AI Explanation**: Groq API (Llama 3.3 70B)
- **Database**: MySQL
- **Frontend**: HTML, CSS, JavaScript, Chart.js

## Project Structure

```
finetunex/
├── main.py                 # FastAPI server, API endpoints
├── trainer.py              # LoRA training loop, model loading, chat
├── explainer.py            # Groq AI explanation of training results
├── database.py             # MySQL connection and training history
├── requirements.txt
├── .env.example
├── .gitignore
├── static/
│   └── index.html          # Frontend UI
├── sample_data/
│   └── customer_support.csv  # Sample training dataset
├── uploads/                # Uploaded datasets (gitignored)
├── trained_adapters/       # Saved LoRA adapters (gitignored)
└── screenshots/
    ├── dataset_upload.png
    ├── training_results.png
    └── chat_interface.png
```

## How It Works

1. Upload a dataset with prompt-completion pairs (CSV or JSON)
2. Select prompt and completion columns, configure LoRA settings
3. Model trains with LoRA — only ~1.4% of parameters are updated instead of all 124M
4. View training loss curve and AI explanation of results
5. Chat with your fine-tuned model and compare against the base model

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key for AI explanations |
| `MYSQL_HOST` | MySQL host (default: localhost) |
| `MYSQL_USER` | MySQL username |
| `MYSQL_PASSWORD` | MySQL password |
| `MYSQL_DATABASE` | Database name (default: finetunex) |

## Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/manojkumar-ra/finetunex.git
cd finetunex
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# edit .env with your API keys and MySQL credentials
```

4. Run the server
```bash
python main.py
```

5. Open `http://localhost:8005` in your browser

## Screenshots

### Dataset Upload & Configuration
![Dataset Upload](screenshots/dataset_upload.png)

### Training Results & Loss Curve
![Training Results](screenshots/training_results.png)

### Chat Interface — Fine-tuned vs Base Model
![Chat Interface](screenshots/chat_interface.png)
