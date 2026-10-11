import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

DOCUMENTS_FOLDER = "documents"


def load_documents(folder_path):
    documents = []

    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return documents

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append((filename, text))

    return documents


# Split documents into paragraph-based chunks
def chunk_text(text, filename):
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []

    for i, para in enumerate(paragraphs):
        chunks.append({
            "id": f"{filename}_chunk_{i}",
            "text": para,
            "source": filename
        })

    return chunks


def build_vector_db():
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

    # Use a different variable name to avoid shadowing the module
    ef = embedding_functions.DefaultEmbeddingFunction()

    # Get the existing collection or create it if necessary
    collection = client.get_or_create_collection(
        name="space_facts",
        embedding_function=ef
    )

    documents = load_documents(DOCUMENTS_FOLDER)
    all_chunks = []

    for filename, text in documents:
        all_chunks.extend(chunk_text(text, filename))

    print(
        f"Loaded {len(documents)} documents "
        f"-> split into {len(all_chunks)} chunks"
    )

    if not all_chunks:
        print("No text chunks found. Check your documents folder.")
        return

    # Insert new chunks or update existing chunks
    collection.upsert(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks]
    )

    print("Vector database built successfully at ./chroma_db")
    print(f"Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    build_vector_db()
