from fastapi import FastAPI
from pydantic import BaseModel

import APIserver.Actions as Actions


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/status/")
def read_item():
    return {"item_id": {ServerStatus}, }



class PredictionRequest(BaseModel):
    data: str

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict")
def predict(request: PredictionRequest):
    # Process the request and make the prediction
    prediction = process_request(request.data)
    # Create the response
    response = PredictionResponse(prediction=prediction)
    return response
