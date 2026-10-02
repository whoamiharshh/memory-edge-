"""Download the offline question-answering model used by Ask (about 130 MB, once).

  .venv\\Scripts\\python.exe -m tools.fetch_qa_model

deepset/roberta-base-squad2 (CC-BY-4.0, trained on SQuAD 2.0 so it can say "this passage does not answer that"), as an
int8-quantized ONNX build from onnx-community/roberta-base-squad2-ONNX, into models_cache/qa/roberta-base-squad2/.
shared/qa.py reads it with ONNX Runtime - no torch. Without it Ask still works, but falls back to the stricter
word-coverage rule and answers fewer Wikipedia questions (see docs/DECISIONS.md D49).
"""
from huggingface_hub import hf_hub_download

REPO = "onnx-community/roberta-base-squad2-ONNX"
FILES = ["onnx/model_quantized.onnx", "tokenizer.json", "config.json", "special_tokens_map.json",
         "tokenizer_config.json", "README.md"]
DEST = "models_cache/qa/roberta-base-squad2"

if __name__ == "__main__":
    for f in FILES:
        print(hf_hub_download(REPO, f, local_dir=DEST), flush=True)
    print("done")
