import uvicorn
from fastapi import FastAPI

from typing import Dict, Any
from pydantic import BaseModel

from typing import Literal
from pydantic import BaseModel, Field

#load pipeline model
import pickle

model_file = 'pipeline_model_C=0.5.bin'
with open(model_file, 'rb') as f_in:
    training_pipeline = pickle.load(f_in) 




#Define input and output schema
#request - pydantic class for request
class Customer(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)
    
    class Config:
        extra = "forbid"


#response -> output
class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


#initiate fastapi
app = FastAPI(title = "churn-prediction-apis")


def predict_single(customer):
    result = training_pipeline.predict_proba(customer)[0,1]
    return float(result)



@app.post("/predict", response_model=PredictResponse)
def predict(customer: Customer) -> PredictResponse:
    churn_proba = predict_single(customer.dict())
    #churn_proba.round(2)

    #action = "send email with promo" if churn_proba >= 0.5 else "Do not send email"
    
    return PredictResponse(
       churn_probability=churn_proba, 
       churn = bool(churn_proba >= 0.5),
       #action = str(action)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

  






# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle

# # Load the saved pipeline model
# model_file = 'pipeline_model_C=0.5.bin'
# with open(model_file, 'rb') as f_in:
#     training_pipeline = pickle.load(f_in)

# # Initialize FastAPI app
# app = FastAPI()

# # Define input schema
# class CustomerData(BaseModel):
#     gender: str
#     seniorcitizen: int
#     partner: str
#     dependents: str
#     phoneservice: str
#     multiplelines: str
#     internetservice: str
#     onlinesecurity: str
#     onlinebackup: str
#     deviceprotection: str
#     techsupport: str
#     streamingtv: str
#     streamingmovies: str
#     contract: str
#     paperlessbiling: str
#     paymentmethod: str
#     tenure: int
#     monthlycharges: float
#     totalcharges: float

# # Define output schema
# class PredictionResult(BaseModel):
#     churn_probability: float
#     action: str

# # Define prediction endpoint
# @app.post("/predict", response_model=PredictionResult)
# def predict_churn(customer: CustomerData):
#     # Convert input to dictionary
#     customer_dict = customer.dict()

#     # Predict churn probability
#     churn_proba = training_pipeline.predict_proba([customer_dict])[0, 1]
#     churn_proba_rounded = round(churn_proba, 2)

#     # Decide action
#     action = "send email with promo" if churn_proba >= 0.5 else "do not send promo"

#     return {
#         "churn_probability": churn_proba_rounded,
#         "action": action
#     }


# # if __name__ == "__main__":
# #     uvicorn.run(app, host = "0.0.0.0", port=9696)