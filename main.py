from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import csv
import os
import json
from trainer import start_training, get_training_status, chat_with_model
from explainer import explain_training
from database import save_run, get_history, init_db

app = FastAPI(title="FineTuneX", version="1.0.0")

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("trained_adapters", exist_ok=True)
init_db()

# keeping session state in memory for now
_session = {}


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # accept csv or json
    if not (file.filename.endswith(".csv") or file.filename.endswith(".json")):
        return {"error": "upload a csv or json file"}

    contents = await file.read()
    filepath = os.path.join("uploads", file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)

    # parse the dataset
    try:
        if file.filename.endswith(".csv"):
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            cols = list(rows[0].keys()) if rows else []
            total = len(rows)
            preview = rows[:5]
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                total = len(data)
                preview = data[:5]
                cols = list(data[0].keys()) if data else []
            else:
                return {"error": "json should be a list of objects"}
    except Exception as e:
        return {"error": f"failed to parse file: {str(e)}"}

    _session["filepath"] = filepath
    _session["filename"] = file.filename
    _session["total"] = total

    return {
        "filename": file.filename,
        "total_examples": total,
        "columns": cols,
        "preview": preview
    }


class TrainRequest(BaseModel):
    prompt_col: str
    completion_col: str
    epochs: int = 3
    lora_rank: int = 8
    learning_rate: float = 0.0003
    batch_size: int = 4


@app.post("/train")
def train(req: TrainRequest):
    if "filepath" not in _session:
        return {"error": "upload a dataset first"}

    result = start_training(
        filepath=_session["filepath"],
        prompt_col=req.prompt_col,
        completion_col=req.completion_col,
        epochs=req.epochs,
        lora_rank=req.lora_rank,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size
    )

    if result.get("error"):
        return {"error": result["error"]}

    # get ai to explain what the training results mean
    result["explanation"] = explain_training(result["summary"])

    save_run(
        filename=_session["filename"],
        base_model=result["summary"]["base_model"],
        total_examples=result["summary"]["total_examples"],
        epochs=req.epochs,
        final_loss=result["summary"]["final_loss"],
        lora_rank=req.lora_rank,
        adapter_path=result["summary"].get("adapter_path", "")
    )

    _session["trained"] = True

    return result


class ChatRequest(BaseModel):
    prompt: str
    use_finetuned: bool = True


@app.post("/chat")
def chat(req: ChatRequest):
    if req.use_finetuned and not _session.get("trained"):
        return {"error": "no fine-tuned model available, train first"}

    response = chat_with_model(
        prompt=req.prompt,
        use_finetuned=req.use_finetuned
    )
    return response


@app.get("/history")
def history():
    return get_history()


@app.get("/status")
def status():
    return get_training_status()


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
