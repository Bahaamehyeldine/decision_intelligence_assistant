import pytest
from pydantic import ValidationError

from schemas.input import QueryRequest


def test_valid_request_defaults_top_k():
    assert QueryRequest(query="my internet is down").top_k == 5


@pytest.mark.parametrize("bad", [{"query": "hi"}, {"query": "x" * 501}, {"query": "valid text", "top_k": 0},
                                 {"query": "valid text", "top_k": 21}])
def test_invalid_requests_rejected(bad):
    with pytest.raises(ValidationError):
        QueryRequest(**bad)
