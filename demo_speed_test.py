import requests
import json
import time

URL_ENCODE = "https://embedding-inversion-demo.jina.ai/encode"
URL_DECODE = "https://embedding-inversion-demo.jina.ai/decode"

HEADERS = {
    # Per superare il blocco CORS descritto nel codice `demo_server.py`
    "Origin": "https://embedding-inversion-demo.jina.ai",
    "Referer": "https://embedding-inversion-demo.jina.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Content-Type": "application/json"
}

def test_inference_speed():
    test_text = "The quick brown fox jumps over the lazy dog"
    print(f"Testando text: '{test_text}'")

    # Step 1: Ottenere l'embedding
    start_encode = time.time()
    resp_enc = requests.post(
        URL_ENCODE, 
        headers=HEADERS, 
        json={"text": test_text, "model": "qwen3"}
    )
    
    if resp_enc.status_code != 200:
        print(f"Errore /encode: {resp_enc.status_code} - {resp_enc.text}")
        return

    emb = resp_enc.json().get("embedding")
    encode_time = time.time() - start_encode
    print(f"[1] Encode completato in: {encode_time:.3f} s")

    # Step 2: Lanciare e misurare la decode in streaming
    print("\n[2] Partenza operazione /decode (Inversion)...")
    start_decode = time.time()
    
    resp_dec = requests.post(
        URL_DECODE, 
        headers=HEADERS, 
        json={"embedding": emb, "steps": 32, "model": "qwen3"},
        stream=True
    )

    if resp_dec.status_code != 200:
        print(f"Errore /decode: {resp_dec.status_code} - {resp_dec.text}")
        return

    tokens_received = False
    tokens_str = ""
    for line in resp_dec.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                try:
                    data = json.loads(decoded_line[6:])
                    
                    if "error" in data:
                        print(f"Server Error in decode stream: {data['error']}")
                        break

                    if "progress" in data and not data.get("done", False):
                        tokens_str = "".join([t["t"] for t in data["tokens"] if t["t"] != "[MASK]"])

                    if data.get("done", False):
                        tokens_received = True
                        break
                except json.JSONDecodeError:
                    pass

    decode_time = time.time() - start_decode
    
    if not tokens_received:
        print("\n[!] Decode non completata o bloccata.")
    else:
        print("Cosine similarity: ", data["cosine_similarity"])
        tokens_str = "".join([t["t"] for t in data["tokens"] if t["t"] != "[MASK]"])
        print(tokens_str)

if __name__ == "__main__":
    test_inference_speed()
