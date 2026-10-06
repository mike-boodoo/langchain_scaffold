# Advanced AI Stack

Production-oriented reference architecture combining:

- **LangChain** — agent orchestration, middleware, structured responses
- **LlamaIndex** — document ingestion, embeddings, retrieval, RAG
- **MCP** — tools, resources, prompts, remote AI capabilities
- **Pydantic v2** — typed contracts and validation
- **FastAPI** — application/API boundary
- **OpenAI** — chat and embedding models

The design separates concerns so the system can evolve from a local AI application into a distributed, multi-service AI platform.

## Architecture

```text
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │       /ask           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     LangChain        │
                         │       Agent          │
                         │                      │
                         │ tools / middleware   │
                         │ structured output    │
                         └───────┬───────┬──────┘
                                 │       │
                        local RAG │       │ MCP
                                 │       │
                                 ▼       ▼
                         ┌──────────┐ ┌──────────┐
                         │LlamaIndex│ │ MCP v2   │
                         │   RAG    │ │ Server   │
                         └────┬─────┘ └────┬─────┘
                              │             │
                              ▼             ▼
                         Documents      Remote tools
                         Embeddings     Resources
                         Vector index   Prompts
                              │
                              └──────┬──────┘
                                     ▼
                              Pydantic v2
                              typed contracts
```

## Repository layout

```text
advanced_ai_stack/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── agent.py
│   ├── config.py
│   ├── index.py
│   ├── main.py
│   ├── mcp_server.py
│   └── schemas.py
├── data/
│   └── example.txt
├── storage/
│   └── .gitkeep
├── tests/
│   └── test_schemas.py
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.11+
- An OpenAI API key
- `uv` recommended, although standard `pip` also works

## Installation

### With uv

```bash
git clone <your-repository-url>
cd advanced_ai_stack

uv venv
source .venv/bin/activate

uv pip install -e '.[dev]'
```

Windows PowerShell:

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e '.[dev]'
```

## Configuration

Copy the example environment:

```bash
cp .env.example .env
```

Then configure:

```env
OPENAI_API_KEY=your-key

OPENAI_MODEL=gpt-5.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

DATA_DIR=./data
PERSIST_DIR=./storage/index

MCP_URL=http://127.0.0.1:8000/mcp
```

Do not commit `.env`.

## Build the knowledge index

Place documents in `data/`.

Then:

```bash
python -m app.index
```

The ingestion pipeline performs:

```text
documents
   ↓
sentence splitting
   ↓
metadata/title extraction
   ↓
embeddings
   ↓
VectorStoreIndex
   ↓
persistent storage
```

## Start the MCP server

In terminal 1:

```bash
python -m app.mcp_server
```

The MCP server exposes:

### Tools

```text
knowledge_search
knowledge_answer
search_books
```

### Resources

```text
kb://health
kb://instructions
```

### Prompts

```text
grounded_research
```

The server uses Streamable HTTP.

## Start the API

In terminal 2:

```bash
uvicorn app.api:app --reload --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

## Ask the agent

```bash
curl \
  -X POST \
  http://127.0.0.1:8080/ask \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: demo' \
  -d '{
    "question": "What does the indexed material say about AI systems?"
  }'
```

The response is validated as a `FinalAnswer` Pydantic model.

Example:

```json
{
  "answer": "The indexed material describes...",
  "confidence": 0.91,
  "citations": [
    {
      "source_id": "abc123",
      "title": "example.txt",
      "excerpt": "Relevant evidence...",
      "score": 0.87,
      "kind": "document"
    }
  ],
  "tool_trace": [],
  "unresolved_questions": []
}
```

## Run the agent directly

```bash
python -m app.main \
  "What does the knowledge base say about typed AI interfaces?"
```

## Tests

```bash
pytest
```

Type checking:

```bash
mypy app
```

Linting:

```bash
ruff check .
```

Formatting:

```bash
ruff format .
```

## Design principles

### 1. Pydantic is the contract layer

Avoid untyped dictionaries between architectural boundaries.

```python
RetrievalQuery
      ↓
LlamaIndex
      ↓
RetrievalResult
      ↓
LangChain
      ↓
FinalAnswer
```

Pydantic provides validation at each boundary.

### 2. LlamaIndex owns knowledge

LlamaIndex is responsible for:

- ingestion
- chunking
- metadata
- embeddings
- indexing
- retrieval
- RAG

LangChain should not need to know how the vector index is implemented.

### 3. MCP owns capability boundaries

MCP is useful when a capability should be:

- remote
- independently deployed
- independently permissioned
- reusable by multiple agents
- implemented in another language/service

For example:

```text
Agent
  │
  ├── local_rag_search
  │
  ├── knowledge__knowledge_search
  │
  ├── database__query
  │
  └── github__issue_search
```

### 4. LangChain owns agent execution

The agent determines when to use:

- local tools
- MCP tools
- retrieval
- model reasoning
- structured output

### 5. Application context is not model-controlled

Tenant IDs, request IDs, permissions, and security context should come from the application/runtime.

Do not ask the model to invent them.

```python
RuntimeContext(
    user_id="...",
    tenant_id="...",
    request_id="...",
)
```

## Security model

This repository is intentionally structured so security controls can be added at boundaries.

Recommended production controls:

```text
HTTP authentication
       ↓
tenant authorization
       ↓
agent runtime context
       ↓
tool authorization
       ↓
MCP server authorization
       ↓
external service
```

Never expose:

- API keys to the model
- database credentials to tool arguments
- arbitrary shell execution
- unrestricted filesystem access
- unrestricted HTTP fetching

Treat tool results as untrusted external data.

## Multi-tenant extension

A production deployment should make the tenant part of retrieval itself.

Conceptually:

```python
RetrievalQuery(
    query="...",
    filters={
        "tenant_id": tenant_id,
    },
)
```

Then enforce the filter in the vector/database layer rather than trusting the LLM to supply it.

## Multi-agent extension

The architecture can grow into:

```text
                       Supervisor
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      Researcher       Programmer        Analyst
          │                │                │
          ▼                ▼                ▼
      LlamaIndex          MCP             MCP
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                       Pydantic
```

The same schemas can become inter-agent contracts.

## Future production upgrades

Recommended next steps:

1. Add PostgreSQL/pgvector or another production vector backend.
2. Add hybrid lexical + vector retrieval.
3. Add reranking.
4. Add LangGraph persistence/checkpointing.
5. Add human approval interrupts for high-risk tools.
6. Add OAuth/API authentication to MCP.
7. Add per-tool authorization policies.
8. Add OpenTelemetry tracing.
9. Add evaluation datasets and regression tests.
10. Add prompt/version management.
11. Add rate limiting and token budgets.
12. Add tenant-scoped indexes.
13. Add asynchronous ingestion workers.
14. Add document lifecycle/version tracking.
15. Add model routing and fallback models.

## Development philosophy

This repository deliberately avoids making the LLM the center of the architecture.

Instead:

```text
                 typed state
                     │
                     ▼
             ┌───────────────┐
             │    Agent      │
             └───────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       retrieval   tools      services
          │          │          │
          └──────────┼──────────┘
                     ▼
              validated state
```

The model proposes actions.

The application validates them.

Tools execute them.

Pydantic validates data.

MCP defines capability boundaries.

LlamaIndex manages knowledge.

LangChain manages agent execution.

That separation makes the system substantially easier to test, secure, observe, and replace piece-by-piece.

## License

MIT
