# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from predict import PredictionPipeline

# Instantiate FastAPI app
app = FastAPI(title="Topic Matching API")

# Load the model pipeline once
pipeline = PredictionPipeline()

# Request schema
class PredictRequest(BaseModel):
    text: str
    top_k: int = 10

# Response schema
class TopicResponse(BaseModel):
    topic_id: str
    topic_text: str
    probability: float

# Endpoint
@app.post("/predict", response_model=List[TopicResponse])
def predict_topics(req: PredictRequest):
    result_df = pipeline.predict(req.text, top_k=req.top_k)
    return result_df[["topic_id", "topic_text", "probability"]].to_dict(orient="records")
