from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware, ToolRetryMiddleware
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from .config import get_settings
from .index import KnowledgeIndex
from .schemas import FinalAnswer, RetrievalQuery, ToolInvocation


@dataclass(frozen=True)
class RuntimeContext:
    user_id: str
    tenant_id: str
    request_id: str


class AIService:
    def __init__(self, context: RuntimeContext) -> None:
        settings = get_settings()
        self.settings = settings
        self.context = context

        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0,
        )

        self.knowledge = KnowledgeIndex()

        self.mcp_client = MultiServerMCPClient(
            {
                "knowledge": {
                    "transport": "streamable_http",
                    "url": settings.mcp_url,
                }
            },
            tool_name_prefix=True,
            handle_tool_errors=True,
        )

    def _local_rag_tool(self):
        @tool
        def local_rag_search(query: str, top_k: int = 6) -> str:
            """Search the local LlamaIndex knowledge base."""
            result = self.knowledge.retrieve(
                RetrievalQuery(query=query, top_k=top_k)
            )
            return result.model_dump_json()

        return local_rag_search

    async def build(self):
        mcp_tools = await self.mcp_client.get_tools()

        tools = [
            *mcp_tools,
            self._local_rag_tool(),
        ]

        system_prompt = f"""
You are an evidence-first AI research agent.

Runtime tenant: {self.context.tenant_id}
Runtime user: {self.context.user_id}
Request ID: {self.context.request_id}

Rules:
1. Prefer knowledge_search or local_rag_search for factual questions.
2. Never claim retrieved evidence says something it does not say.
3. Distinguish evidence, synthesis, inference, and uncertainty.
4. Use MCP tools for remotely governed capabilities.
5. Keep tool arguments minimal.
6. Respect every schema.
7. Produce the final answer using FinalAnswer.
"""

        return create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
            response_format=FinalAnswer,
            middleware=[
                ToolRetryMiddleware(
                    max_retries=2,
                    backoff_factor=1.5,
                    initial_delay=0.5,
                    max_delay=4.0,
                    jitter=True,
                ),
                ToolErrorMiddleware(),
            ],
            debug=False,
            name="evidence_first_agent",
        )

    async def invoke(self, question: str) -> FinalAnswer:
        agent = await self.build()

        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            }
        )

        structured = result.get("structured_response")

        answer = (
            structured
            if isinstance(structured, FinalAnswer)
            else FinalAnswer.model_validate(structured)
        )

        answer.tool_trace.append(
            ToolInvocation(
                tool_name="agent",
                arguments={"question": question},
                purpose="orchestrated research",
                risk="low",
            )
        )

        return answer

    async def stream(self, question: str):
        agent = await self.build()

        async for event in agent.astream(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            stream_mode="updates",
        ):
            yield event
