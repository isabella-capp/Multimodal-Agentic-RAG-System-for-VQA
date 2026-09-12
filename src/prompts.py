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

# Braces are doubled so the JSON survives `.format()` on the templates below.
_FORMAT_LITERAL = ANSWER_FORMAT.replace("{", "{{").replace("}", "}}")


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

NO_RAG_PROMPT = f"""\
Answer the question about the image.

--- QUESTION ---
{{question}}

{_FORMAT_LITERAL}"""

RAG_PROMPT = f"""\
Answer the question using the image and the context below. Use only information \
that is in the context or visible in the image.

--- CONTEXT ---
{{context}}

--- QUESTION ---
{{question}}

{_FORMAT_LITERAL}"""

NO_RAG_PROMPT_LEGACY = "{question}"

RAG_PROMPT_LEGACY = """\
Answer the question concisely based on the provided image and the following \
context. Strictly use only the information provided in the context or visible \
in the image.

--- CONTEXT ---
{context}

--- QUESTION ---
{question}

"""

RAG_PROMPT_DIRECT = """\
Answer the question concisely based on the provided image and the following \
context. Strictly use only the information provided in the context or visible \
in the image.

Start with the answer itself. Do not open with "Based on the context", "The \
image shows" or any similar preamble. If the context gives a range, several \
values or several places, give all of them as the context states them, with the \
same units and wording.

--- CONTEXT ---
{context}

--- QUESTION ---
{question}

"""
