from rag_tutorials.chunking import fixed_chunk_documents, semantic_chunk_documents
from rag_tutorials.data_generation import generate_documents, generate_queries


if __name__ == "__main__":
    docs = generate_documents(doc_count=10)
    queries = generate_queries(docs, query_count=5)
    fixed = fixed_chunk_documents(docs)
    semantic = semantic_chunk_documents(docs)
    print(
        {
            "docs": len(docs),
            "queries": len(queries),
            "fixed_chunks": len(fixed),
            "semantic_chunks": len(semantic),
        }
    )
