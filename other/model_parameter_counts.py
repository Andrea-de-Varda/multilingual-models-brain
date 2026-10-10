"""
Parameter counts of the 20 multilingual models (Table 1), read from the Hugging Face checkpoints.
"""
import torch
from transformers import AutoConfig, AutoModel

# List of model names
model_names = [
    "facebook/xglm-4.5B",
    "facebook/xglm-2.9B",
    "facebook/xglm-564M",
    "facebook/xglm-1.7B",
    "bert-base-multilingual-cased",
    "distilbert-base-multilingual-cased",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "google/mt5-small",
    "google/mt5-base",
    "google/mt5-large",
    "microsoft/mdeberta-v3-base",
    "microsoft/xlm-align-base",
    "microsoft/infoxlm-base",
    "microsoft/infoxlm-large",
    "microsoft/Multilingual-MiniLM-L12-H384",
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-distilled-1.3B",
    "facebook/nllb-200-1.3B",
    "ai-forever/mGPT"
]

for model_name in model_names:
    # Build the model on PyTorch's "meta" device: only the configuration is downloaded, no weights.
    with torch.device("meta"):
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
    total = model.num_parameters()
    line = f"{model_name:45s} {total / 1e6:8.1f}M parameters"
    if hasattr(model, "get_encoder"):  # encoder-decoder models (mT5, NLLB): only the encoder is used in the paper
        encoder = sum(p.numel() for p in model.get_encoder().parameters())
        line += f"  (encoder only, incl. shared embeddings: {encoder / 1e6:.1f}M)"
    print(line)
