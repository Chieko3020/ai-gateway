"""
本地 Embedding 服务：OpenAI 兼容 /v1/embeddings
使用 ONNX Runtime（轻量，无需 PyTorch，~150MB 运行时）
模型路径：项目根目录/bge-small-zh-v1.5-onnx/
启动: /tmp/onnx-venv/bin/python scripts/embed_server.py
"""
import json
import os
import sys
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "bge-small-zh-v1.5-onnx")


class EmbedHandler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None

    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        text = body.get("input", "")
        if isinstance(text, list):
            text = text[0]

        # tokenize → ONNX inference → mean pooling → normalize
        inputs = self.tokenizer(text, return_tensors="np",
                                 padding=True, truncation=True)
        outputs = self.model.run(None, dict(inputs))[0]

        # Mean pooling over token dimension (last_hidden_state mean)
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, axis=-1)
        pooled = np.sum(outputs * mask_expanded, axis=1) / np.clip(
            np.sum(mask_expanded, axis=1), 1e-9, None)

        # L2 normalize
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        vec = pooled[0].tolist()

        resp = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": vec}],
            "model": "bge-small-zh-v1.5-onnx",
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):
        print(f"[embed] {args[0]}", file=sys.stderr)


def main():
    import onnxruntime as ort
    from transformers import AutoTokenizer

    print(f"[embed] loading ONNX model from {MODEL_PATH}...", file=sys.stderr)
    onnx_file = os.path.join(MODEL_PATH, "model.onnx")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2  # 限制 CPU 线程
    EmbedHandler.model = ort.InferenceSession(onnx_file, sess_opts,
                                               providers=["CPUExecutionProvider"])
    EmbedHandler.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("[embed] ONNX model loaded", file=sys.stderr)

    server = HTTPServer(("127.0.0.1", 8081), EmbedHandler)
    print("[embed] listening on :8081", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
