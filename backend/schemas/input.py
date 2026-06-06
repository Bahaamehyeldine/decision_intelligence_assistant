"""
input.py
--------
Pydantic schemas for API request validation.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for /query endpoint request."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User support query text"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar tickets to retrieve"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "my internet has been down for 3 days",
                "top_k": 5
            }
        }
    }