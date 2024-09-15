
from fastapi import FastAPI, HTTPException
from app.funcionalidad1 import usar_infoSE
from app.funcionalidad2 import usar_infoSE_2
from app.funcionalidad3 import funcionalidad3
from pydantic import BaseModel
from typing import List

app = FastAPI()

class SingleParamModel(BaseModel):
    param: int

class DoubleParamsModel(BaseModel):
    param1: int
    param2: int

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
    result = funcionalidad3(params.params)
    return {"result": result}