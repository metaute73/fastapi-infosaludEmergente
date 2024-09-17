
from fastapi import FastAPI, HTTPException
from app.funcionalidad1 import usar_infoSE
from app.funcionalidad2 import usar_infoSE_2
#from app.funcionalidad3 import usar_infoSE3, Model
from pydantic import BaseModel
from typing import List
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5137",  # Allow localhost:5137
]

# Add the CORSMiddleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Specify allowed origins
    allow_credentials=True,          # Allow cookies and credentials
    allow_methods=["*"],             # Allow all HTTP methods
    allow_headers=["*"],             # Allow all headers
)

class Prediction(nn.Module):
    def __init__(self, in_features=9, h1=24, h2=32, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

def usar_infoSE3(respuestas_usuario):
    
    modelito = Prediction()
    modelito = torch.load('model.pth')
    if modelito.forward(torch.FloatTensor(respuestas_usuario)).argmax().item() == 0:
        return 'Por ahora no pareces estar en riesgo'
    return 'Podrías estar en riesgo, te recomendamos realizarte una prueba de VIH lo más pronto posible'
  

class SingleParamModel(BaseModel):
    param: str

class DoubleParamsModel(BaseModel):
    param1: int
    param2: str

class ListParamModel(BaseModel):
    params: List[int]

@app.post("/process_single")
async def process_single(params: SingleParamModel):
    result = usar_infoSE(params.param)
    return {"result": result}

@app.post("/process_double")
async def process_double(params: DoubleParamsModel):
    result = usar_infoSE_2(params.param1, params.param2)
    return {"result": result}

@app.post("/process_list")
async def process_list(params: ListParamModel):
    result = usar_infoSE3(params.params)
    return {"result": result}

        
