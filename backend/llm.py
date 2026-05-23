import ollama

def generate_answer(query, chunks):

    # Combine retrieved chunks
    context = "\n\n".join([
        chunk["text"] for chunk in chunks
    ])

    # Generic RAG Prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the provided context.

Guidelines:
- Do not make up information
- If the answer is not present in the context, say:
  "I could not find the answer in the provided documents."
- Keep the answer clear and concise
- Use bullet points when appropriate
- Focus only on relevant information from the context

Context:
{context}

Question:
{query}
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