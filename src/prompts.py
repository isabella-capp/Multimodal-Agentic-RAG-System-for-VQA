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

# The format asks for JSON, so the braces must survive `.format()` on the
# templates below: Python reads `{"answer"}` as a replacement field and raises
# KeyError('"answer"') on every example. Doubling them makes them literal.
_FORMAT_LITERAL = ANSWER_FORMAT.replace("{", "{{").replace("}", "}}")


def extract_answer(raw: str) -> str:
    """The `answer` field of the model's JSON reply, or the reply as it came.

    Shared by A, B and the agent on purpose. The extraction used to live only in
    the agent, so the baselines scored the raw JSON against the gold answer —
    the settings must agree on what counts as an answer or their scores are not
    comparable.
    """
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


# A — no retrieval: the model answers from the image and its own knowledge.
NO_RAG_PROMPT = f"""\
Answer the question about the image.

--- QUESTION ---
{{question}}

{_FORMAT_LITERAL}"""

# B — retrieval: the model answers from the image and the retrieved paragraphs.
RAG_PROMPT = f"""\
Answer the question using the image and the context below. Use only information \
that is in the context or visible in the image.

--- CONTEXT ---
{{context}}

--- QUESTION ---
{{question}}

{_FORMAT_LITERAL}"""


# The prompts as they were before the shared format constraint. Kept to
# reproduce the historical B=0.401: that run averaged 19.3 words per answer
# against 2.4 now, and BEM rewards the extra surface — the correct answer is
# contained just as often either way (0.308 vs 0.297). Running B with these
# separates "the metric liked long answers" from "the model knew less".
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


# The long-form prompt, minus the preambles. Answers that open with "Based on
# the provided context, the insect in the image..." appeared in three of ten
# sampled templated failures: the answer is in there, buried, and BEM scores the
# whole string. This asks for the same free-form length — which BEM rewards, and
# which the answer-format block gives up — without the throat-clearing, and asks
# for the full range when the passage gives one, since a gold answer of
# "63 to 65 days or 54 to 70 days" is missed by a reply of "63 to 65 days".
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
