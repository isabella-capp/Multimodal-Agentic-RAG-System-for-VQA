from __future__ import annotations

import base64
import mimetypes

from langchain_core.messages import HumanMessage


def image_to_data_uri(image_path: str) -> str:
    """Encode an image file as a base64 ``data:`` URI for an image_url block."""
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def build_user_message(image_path, question) -> HumanMessage:
    """First turn: the question text plus the query image."""
    return HumanMessage(
        content=[
            {
                "type": "text",
                "text": question
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(image_path)}
            },
        ]
    )
