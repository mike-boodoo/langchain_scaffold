from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .agent import AIService, RuntimeContext
from .schemas import FinalAnswer, Health

app = FastAPI(
    title="Advanced AI Stack",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=8_000)


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(
        component="api",
        details={"status": "ready"},
    )


@app.post("/ask", response_model=FinalAnswer)
async def ask(
    request: AskRequest,
    x_tenant_id: str = Header(default="demo-tenant"),
) -> FinalAnswer:
    context = RuntimeContext(
        user_id="api-user",
        tenant_id=x_tenant_id,
        request_id=str(uuid4()),
    )

    service = AIService(context)

    try:
        return await service.invoke(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
