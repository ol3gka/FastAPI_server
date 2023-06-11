import joblib
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

# App creation and model loading
app = FastAPI()
model = joblib.load("./model.joblib")

class Data(BaseModel):
    #X: list
    X: List[List[float]]

@app.get("/")
async def root():
   return {"message": "Its Fast API for ML model deploy"}

@app.post('/predict')
def predict(data: Data):
    """
    :data: input data from the post request
    :return: predicted array for test data
    """
    prediction = model.predict(data.X)
    return {"prediction": prediction.tolist()}

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
