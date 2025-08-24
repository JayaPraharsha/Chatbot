from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.graph import answer_query

app = FastAPI(title="rag_wixqa")
router = APIRouter()


class Query(BaseModel):
    query: str


@router.post("/answer_user_query")
def answer_user_query(query: Query):
    answer = answer_query(query.query)["response"]
    return {"answer": answer}


@router.get("/health")
def health_check():
    return JSONResponse(status_code=200, content={"message": "Everything is good here 👀"})


app.include_router(router)
