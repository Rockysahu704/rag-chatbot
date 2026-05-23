import ollama


def generate_answer(query, chunks):

    # No retrieved chunks
    if not chunks:
        return "I could not find the answer in the provided documents."

    # Build context with metadata
    context = "\n\n".join([
        f"""
Document: {chunk.get("filename", "Unknown")}
Page: {chunk.get("page", "N/A")}

Content:
{chunk["text"]}
"""
        for chunk in chunks
    ])

    # Improved RAG prompt
    prompt = f"""
You are an intelligent RAG assistant.

You must answer ONLY from the provided context.

STRICT RULES:
- Do NOT invent information
- Do NOT assume anything
- If the answer is missing, say:
  "I could not find the answer in the provided documents."
- If user asks about:
  • technologies
  • frameworks
  • libraries
  • tools
  • tech stack
  then ONLY extract exact names from context
- Keep answers concise
- Use bullet points when suitable
- Ignore unrelated context
- Prefer exact factual extraction over explanation

-------------------------
CONTEXT:
{context}
-------------------------

QUESTION:
{query}

ANSWER:
"""

    try:

        response = ollama.chat(
            model="tinyllama",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"Error generating response: {str(e)}"