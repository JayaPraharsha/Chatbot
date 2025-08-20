from fastapi import APIRouter, FastAPI
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


app.include_router(router)

# if __name__ == "__main__":
#     print("Starting server...")
#     uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
