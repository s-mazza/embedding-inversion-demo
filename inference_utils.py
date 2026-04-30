import json
import requests

JINA_HEADERS = {
    "Origin": "https://embedding-inversion-demo.jina.ai",
    "Referer": "https://embedding-inversion-demo.jina.ai/",
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
}
URL_ENCODE = "https://embedding-inversion-demo.jina.ai/encode"
URL_DECODE = "https://embedding-inversion-demo.jina.ai/decode"

def invert_text(text: str) -> str:
    """Encode text to embedding via Jina demo, then decode back to text."""
    resp_enc = requests.post(URL_ENCODE, headers=JINA_HEADERS,
                             json={"text": text, "model": "qwen3"})
    if resp_enc.status_code != 200:
        raise RuntimeError(f"Encode failed ({resp_enc.status_code}): {resp_enc.text}")
    emb = resp_enc.json()["embedding"]

    resp_dec = requests.post(URL_DECODE, headers=JINA_HEADERS,
                             json={"embedding": emb, "steps": 32, "model": "qwen3"},
                             stream=True)
    if resp_dec.status_code != 200:
        raise RuntimeError(f"Decode failed ({resp_dec.status_code}): {resp_dec.text}")

    for line in resp_dec.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                try:
                    data = json.loads(decoded[6:])
                    if data.get("done"):
                        return data["text"]
                except Exception:
                    pass
    return ""
