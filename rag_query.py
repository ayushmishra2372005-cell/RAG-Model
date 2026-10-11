
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def retrieve_relevant_chunks(query, n_results=3):
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_collection(
        name="space_facts",
        embedding_function=embedding_function
    )

    # Avoid requesting more results than the collection contains
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count)
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    sources = [
        metadata.get("sources", "Unknown source")
        if metadata else "Unknown source"
        for metadata in metadatas
    ]

    return list(zip(chunks, sources, distances))


def build_augmented_prompt(query, retrieved_chunks):
    context_text = "\n\n".join(
        f"Source: {source}\n{chunk}"
        for chunk, source, _ in retrieved_chunks
    )

    prompt = f"""You are a helpful assistant.
Answer the user's question using only the context provided below.

If the answer is not contained in the context, say:
"I don't have enough information to answer that question."

Context:
{context_text}

User's question:
{query}

Answer:"""

    return prompt


def generate_answer(prompt):
    try:
        import ollama

        response = ollama.chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except ImportError:
        print("Error: Install the Ollama Python package using pip.")
        return None

    except Exception as e:
        print(f"Ollama error: {e}")
        return None


def main():
    query = input("Ask a question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    try:
        retrieved_chunks = retrieve_relevant_chunks(query)

        if not retrieved_chunks:
            print("No relevant documents found in the collection.")
            return

        prompt = build_augmented_prompt(query, retrieved_chunks)
        answer = generate_answer(prompt)

        if answer:
            print("\nAnswer:")
            print(answer)
        else:
            print("Unable to generate an answer.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
