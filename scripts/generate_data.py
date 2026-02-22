from rag_tutorials.data_generation import build_and_save_dataset


def main() -> None:
    """Generate and persist canonical handbook-derived tutorial datasets."""
    build_and_save_dataset(output_dir="data", doc_count=500, query_count=200)
    print("Generated data/documents.jsonl and data/queries.jsonl")


if __name__ == "__main__":
    main()
