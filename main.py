import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from src.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()
    pipeline.load_and_index()

    print("\nRAG Assistant Ready (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        response = pipeline.query(query)

        print("\nAssistant:", response["answer"])

        if response["sources"]:
            print("\nSources:")
            for s in response["sources"]:
                print(f"- {s['source']} (page {s.get('page')})")

        print()


if __name__ == "__main__":
    main()