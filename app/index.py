from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import (
    Settings as LlamaSettings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .config import get_settings
from .schemas import RetrievedChunk, RetrievalQuery, RetrievalResult

logger = logging.getLogger(__name__)


class KnowledgeIndex:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings

        self.llm = OpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        self.embed_model = OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )

        LlamaSettings.llm = self.llm
        LlamaSettings.embed_model = self.embed_model

        self.index: VectorStoreIndex | None = None

    def build(self, data_dir: Path | None = None) -> VectorStoreIndex:
        source_dir = data_dir or self.settings.data_dir
        documents = SimpleDirectoryReader(
            str(source_dir),
            recursive=True,
        ).load_data()

        if not documents:
            raise ValueError(f"No documents found under {source_dir}")

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=700, chunk_overlap=120),
                TitleExtractor(nodes=3),
                self.embed_model,
            ]
        )

        nodes = pipeline.run(
            documents=documents,
            show_progress=True,
        )

        self.index = VectorStoreIndex(nodes)
        self.index.storage_context.persist(
            persist_dir=str(self.settings.persist_dir)
        )

        logger.info(
            "Indexed %s documents into %s nodes",
            len(documents),
            len(nodes),
        )
        return self.index

    def load(self) -> VectorStoreIndex:
        if self.index is not None:
            return self.index

        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.settings.persist_dir)
        )
        self.index = load_index_from_storage(storage_context)
        return self.index

    def ensure(self) -> VectorStoreIndex:
        try:
            return self.load()
        except Exception:
            logger.info("No persisted index available; rebuilding")
            return self.build()

    def retrieve(self, request: RetrievalQuery) -> RetrievalResult:
        index = self.ensure()
        retriever = index.as_retriever(similarity_top_k=request.top_k)
        results = retriever.retrieve(request.query)

        chunks: list[RetrievedChunk] = []

        for item in results:
            score = float(item.score) if item.score is not None else None

            if (
                request.min_score is not None
                and (score is None or score < request.min_score)
            ):
                continue

            metadata = {
                str(k): str(v)
                for k, v in item.node.metadata.items()
                if v is not None
            }

            chunks.append(
                RetrievedChunk(
                    node_id=item.node.node_id,
                    text=item.node.get_content(),
                    title=metadata.get("document_title")
                    or metadata.get("title"),
                    score=score,
                    metadata=metadata,
                )
            )

        return RetrievalResult(
            query=request.query,
            chunks=chunks,
        )

    def query(self, request: RetrievalQuery) -> str:
        index = self.ensure()
        query_engine = index.as_query_engine(
            similarity_top_k=request.top_k,
            response_mode="compact",
        )
        return str(query_engine.query(request.query))


def main() -> None:
    KnowledgeIndex().build()


if __name__ == "__main__":
    main()
