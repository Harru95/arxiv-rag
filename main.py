import os
from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from rag import build_index,query

app=FastAPI()

app.mount("/static",StaticFiles(directory="static"),name="static")
templates=Jinja2Templates(directory="templates")

index_store={"index":None}

#req/res models

class IndexRequest(BaseModel):
    urls:list[str]

class QueryRequest(BaseModel):
    question: str

#Routes

@app.get("/")

def home(request:Request):
    """serve frontend"""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/index")
def create_index(data:IndexRequest):
    """Recieve URLs,build FAISS,store in memory"""
    try:
        index_store["index"]=build_index(data.urls)
        return {"message":f"Successfully indexed {len(data.urls)} paper(s)"}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.post("/query")
def  answer_question(data:QueryRequest):
    """Recieve ques,query index,ret ans with citations"""
    if index_store["index"] is None:
        raise HTTPException(
            status_code=400,
            detail="No papers indexed yet.Please index papers first."
        )
    try:
        result=query(index_store["index"],data.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))