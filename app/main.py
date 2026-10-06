from __future__ import annotations

import asyncio
import sys

from .agent import AIService, RuntimeContext


async def run(question: str) -> None:
    service = AIService(
        RuntimeContext(
            user_id="cli-user",
            tenant_id="cli",
            request_id="cli-request",
        )
    )

    answer = await service.invoke(question)

    print(answer.model_dump_json(indent=2))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python -m app.main "your question"'
        )

    asyncio.run(run(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
