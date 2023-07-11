import util as utils
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import data_pipeline as data_pipeline
import preprocessing as preprocessing

config = utils.load_config()
model_data = utils.pickle_load(config["production_model_path"])

class api_data(BaseModel):
    stasiun : int
    pm10 : int
    pm25 : int
    so2 : int
    co : int
    o3 : int
    no2 : int
    max : int
    critical : int

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)  # type: ignore
    data.columns = config["predictors"]

    # Convert dtype
    data = data.astype(int)

    # Check range data
    try:
        data_pipeline.check_data(data, config, True)  # type: ignore
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}

    # Predict data
    y_pred = model_data.predict(data)
    label = [0,1]
    predict = model_data.predict(data)

    if y_pred[0] is None:
        y_pred = "Tidak ada API"
    else:
        y_pred = "Ada API"
    return {"res" : y_pred, "error_msg": "", "prediction" : label[predict[0]]}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
