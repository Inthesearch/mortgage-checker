import requests, json
from src.chunker import chunk_files


def embed_text(text):
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
                },
            timeout=10
        )

        response.raise_for_status()
        return response.json().get("embedding")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not responding, please restart and try again later.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Request timed out, please try again.")
    except requests.exceptions.HTTPError:
        raise RuntimeError("Ollama rejected the request. Check that the model name is spelled correctly.")


def build_records(source_list):
    list_with_embeddings = []
    for source in source_list:
        data=dict(source)
        data["embedding"]=embed_text(source["text"])
        list_with_embeddings.append(data)
    return list_with_embeddings

def save_index(records, path ):
    with open(path, "w") as f:
        json.dump(records, f)

def load_index(path):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
        return records


if __name__ == "__main__":
    record = build_records(chunk_files("osfi_mqr"))
    record.extend(build_records(chunk_files("cmhc_requirements")))
    save_index(record, "data/rules_index.json")
    record_list=load_index("data/rules_index.json")
    # vector = embed_text("hello world")
    print(len(record_list))