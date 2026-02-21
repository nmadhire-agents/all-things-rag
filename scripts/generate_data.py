from rag_tutorials.data_generation import build_and_save_dataset


if __name__ == "__main__":
    build_and_save_dataset(output_dir="data", doc_count=500, query_count=200)
    print("Generated data/documents.jsonl and data/queries.jsonl")
