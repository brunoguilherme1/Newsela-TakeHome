from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from predict import PredictionPipeline

app = FastAPI(title="Topic Matching API")

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model pipeline once
pipeline = PredictionPipeline()

# Request and Response Schemas
class PredictRequest(BaseModel):
    text: str
    top_k: int = 10

class TopicResponse(BaseModel):
    topic_id: str
    topic_text: str
    probability: float

# Serve index.html on root
@app.get("/", response_class=HTMLResponse)
def serve_home():
    return FileResponse("static/index.html")

# POST endpoint for predictions
@app.post("/predict", response_model=List[TopicResponse])
def predict_topics(req: PredictRequest):
    result_df = pipeline.predict(req.text, top_k=req.top_k)
    return result_df[["topic_id", "topic_text", "probability"]].to_dict(orient="records")
