from transformers import AutoModel
import gc

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
    # Load the model
    model = AutoModel.from_pretrained(model_name)
    
    # Get the number of parameters
    num_parameters = model.num_parameters()
    
    print(f"The number of parameters in the {model_name} model is: {num_parameters}")
    del model
    gc.collect()
    
    
    
