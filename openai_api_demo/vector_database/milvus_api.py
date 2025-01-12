import os

import torch
import uvicorn
import milvus_service as milvus_service


from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# define InsertRequest structure
class InsertRequest(BaseModel):
    collection_name: str
    data: List[str]


# define InsertResponse structure
class InsertResponse(BaseModel):
    status: str
    message: str


@app.post("/insert_data", response_model=InsertResponse)
async def insert_data(request: InsertRequest):
    milvus_service.insert_data(request.collection_name, request.data)
    response = {
        "status": "success",
        "message": "Data inserted successfully",
    }
    return response


if __name__ == "__main__":
    milvus_service.init()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
