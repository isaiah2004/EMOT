from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

import src.APIserver.Actions as Actions

import sys
print(sys.path)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/status/")
def read_item():
    ServerStatus=CheckServerStatus()
    return {"status": {str(ServerStatus)}, }

class PredictionRequest(BaseModel):
    data: None

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict")
def predict(request: PredictionRequest):
    # Process the request and make the prediction
    prediction = process_request(request.data)
    # Create the response
    response = PredictionResponse(prediction=prediction)
    return response


def CheckServerStatus():
    actionClassifierStatus  = Actions.ActionOnlineStatus
    emotionClassifierStatus = Actions.ActionOnlineStatus

    return [actionClassifierStatus,emotionClassifierStatus]

def process_request(data):
    prediction = Actions.predict_action(data)
    return prediction