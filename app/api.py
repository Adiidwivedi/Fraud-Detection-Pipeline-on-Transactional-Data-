"""FastAPI service.   uvicorn app.api:app --port 8000"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraudpipe.scoring import Scorer

state = {}


@asynccontextmanager
async def lifespan(_):
    state["scorer"] = Scorer()
    yield


app = FastAPI(title="Fraud Detection API", version="1.0", lifespan=lifespan)


class Transaction(BaseModel):
    step: int = Field(ge=0, description="hour index since start of simulation")
    type: Literal["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]
    amount: float = Field(ge=0)
    nameOrig: str
    oldbalanceOrg: float = Field(ge=0)
    nameDest: str
    oldbalanceDest: float = Field(ge=0)
    orig_tx_count_24h: Optional[int] = 0
    orig_time_since_last: Optional[int] = 10_000
    dest_tx_count_24h: Optional[int] = 0
    dest_time_since_last: Optional[int] = 10_000


@app.get("/health")
def health():
    return {"status": "ok", "threshold": state["scorer"].threshold}


@app.post("/predict")
def predict(tx: Transaction, explain: bool = True):
    try:
        return state["scorer"].score(tx.model_dump(), explain=explain)
    except Exception as e:  # pragma: no cover
        raise HTTPException(500, str(e))
