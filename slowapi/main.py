from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from toxic_comments.predict_model import predict_user_input, predict_file_input
from omegaconf import OmegaConf

app = FastAPI()

class PredictTextInput(BaseModel):
    text: str

class PredictFileInput(BaseModel):
    file: str

@app.post("/predict/user_input")
async def predict_user_input_endpoint(item: PredictTextInput):
    config = OmegaConf.load("toxic_comments/models/config/default.yaml")  # Update with the actual path
    config.text = item.text
    predict_user_input(config)
    return JSONResponse(content={"message": "Prediction completed!"})


@app.post("/predict/file_input")
async def predict_file_input_endpoint(item: PredictFileInput):
    config = OmegaConf.load("models/config/default.yaml")  # Update with the actual path
    config.file = item.file
    predict_file_input(config)
    return JSONResponse(content={"message": "Prediction completed!"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
