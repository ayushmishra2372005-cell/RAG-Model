def build_vector_db():
    client=chromadb.PersistentClient(path="./chroma_db",settings=Settings(anonymized_telemetry=False))
    embedding_functions=embedding_functions.DefaultEmbeddingFunction()
    collection=client.create_collection(name="space_facts",embedding_function=embedding_functions)
    documents=load_documents(DOCUMENTS_FOLDER)
    all_chunks=[]
    for filename,text in documents:
        all_chunks.extend(chunk_text(text,filename))
    print(f"Loaded{len(documents)}documents->slit into{len(all_chunks)}chunks")

    collection.add(ids=[c["id"]for c in all_chunks],documents=[c["text"]for c in all_chunks],metadatas=[{"source":c["source"]}for c in all_chunks])
    print("Vector databse built successfully at./chroma_db")
    print(f"Total chunks stored: {collection.count()}")