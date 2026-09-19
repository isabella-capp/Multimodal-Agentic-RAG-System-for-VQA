import json

ANSWER_FORMAT = """\
CRITICAL: When you are ready to answer the user's question, your FINAL message MUST \
be ONLY a valid JSON object. Do not include any other text, no markdown blocks, no \
explanations, no "Here is the answer".

Use exactly this format:
{"answer": "your concise answer here"}

Example 1: {"answer": "1889"}
Example 2: {"answer": "copper, zinc"}
"""


def extract_answer(raw: str) -> str:
    """The `answer` field of the model's JSON reply, or the reply as it came."""
    if not raw:
        return raw
    text = raw.replace("```json", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1]).get("answer", raw)
        except (json.JSONDecodeError, AttributeError):
            pass
    return raw

NO_RAG_PROMPT = "{question}"

RAG_PROMPT = """\
Answer the question concisely based on the provided image and the following \
context. Strictly use only the information provided in the context or visible \
in the image.

--- CONTEXT ---
{context}

--- QUESTION ---
{question}

"""
