from pydantic import ValidationError
import pytest

from app.schemas import FinalAnswer, RetrievalQuery


def test_retrieval_query():
    query = RetrievalQuery(
        query="  distributed   systems  ",
        top_k=5,
    )

    assert query.query == "distributed   systems"


def test_rejects_empty_query():
    with pytest.raises(ValidationError):
        RetrievalQuery(query="")


def test_final_answer_contract():
    result = FinalAnswer(
        answer="A typed response.",
        confidence=0.95,
    )

    assert result.confidence == 0.95


def test_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        FinalAnswer(
            answer="bad",
            confidence=2.0,
        )
