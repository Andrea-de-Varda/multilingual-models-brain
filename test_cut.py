

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.2-1B" # "Qwen/Qwen2.5-1.5B" # "meta-llama/Llama-3.2-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

# Prompt
prompts = ["twenty-seven plus thirty-six equals", 
           "forty-six minus thirty-nine equals",
           "seventy-two plus ninety-five equals"]

prompts = [f"""Let's do some verbal arithmetic: {prompt}""" for prompt in prompts]

for add_space in [True, False]:
    for prompt in prompts:
        print(prompt, f" || add space = {add_space}")
        if add_space:
            inputs = tokenizer(prompt+" ", return_tensors="pt").to(device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Get logits for next token
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # next-token logits
        # Top 5 predictions
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=5)
        top_ids = topk.indices[0].tolist()
        top_tokens = tokenizer.convert_ids_to_tokens(top_ids)
        top_scores = topk.values[0].tolist()
        # Print
        for tok, score in zip(top_tokens, top_scores):
            print(f"{tok}: {score:.4f}")


for add_space in [True, False]:
    for prompt in prompts:
        print(prompt, f" || add space = {add_space}")
        input_text = prompt + " " if add_space else prompt
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        generated_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("continuation:", generated_text)



# FEW-SHOT
fewshot = (
    "Convert the result to words only.\n"
    "Example: sixteen plus nine equals twenty‑five\n"
    "Example: ninety minus ten equals eighty\n")

fewshot = """These are correct arithmetic expressions in English words only, lowercase, no digits or punctuation.
twenty‑seven plus thirty‑six equals"""

problems = [
    "twenty‑seven plus thirty‑six equals", # 63
    "forty‑six minus thirty‑nine equals",  # 7
    "seventy‑two plus ninety‑five equals"  # 167
]

for p in problems:
    prompt = fewshot + "Question: " + p + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).lstrip()
    print(p, "→", answer.split()[0])  # keep first token(s) as needed