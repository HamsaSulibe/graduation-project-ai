from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Graduation Project AI API")

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # TODO: load fine-tuned embedding model + retrieve best cards
    return {"input": req.text, "prediction": "placeholder"}