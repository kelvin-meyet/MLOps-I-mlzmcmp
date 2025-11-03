from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI(title = "ping")

@app.get("/ping")
def ping():
    return "ponging kelvin"

# #request schema
# class InputData(BaseModel):
#     x: int
    
# #response schema
# class OutputData(BaseModel):
#     result: int

# #endpoint
# @app.post("/square", response_model=OutputData)
# def compute_square(data: InputData):
#     result = data.x * data.x
#     return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=9696)