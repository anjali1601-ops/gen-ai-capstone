"""CLI alias of the same batch workflow as main.py."""

from src import config
from src.llm import LLMClient
from src.pipeline import process_folder


def main() -> None:
    config.reload_env()
    client = LLMClient()
    print(f"Provider: {client.provider} | Model: {client.model}")
    results = process_folder(config.DATA_DIR, config.OUTPUT_DIR, client)
    processed = sum(1 for item in results if item.status == "processed")
    print(f"Complete. {processed}/{len(results)} document(s) processed.")
    print(f"Report: {config.OUTPUT_DIR / 'final_report.csv'}")


if __name__ == "__main__":
    main()
