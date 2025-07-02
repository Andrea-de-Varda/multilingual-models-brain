#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:51:56 2024

@author: dev
"""



# NOTE FOR MYSELF: with previous method (results/monolingual_{model_prefix}), correlation is positive
def load_monol(model_prefix):
    with open(f"results/monolingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def load_multi(model_prefix):
    with open(f"results/multilingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

out = []
for model in model_names:
    multi_data = load_multi(model)
    monol_data = load_monol(model)
    for layer in multi_data.keys():
        mul = multi_data[layer]
        mon = monol_data[layer]
        merged = pd.merge(mul, mon)
        merged["layer"] = layer
        merged["model"] = model
        out.append(merged)
out = pd.concat(out)

# for model in model_names:
#     temp = out[out.model == model]
#     plt.figure(dpi=300, figsize=(3, 2))
#     plt.scatter(temp["m"], temp["r"], c=temp["layer"])
#     plt.xlabel("Monolingual performance")
#     plt.ylabel("Transfer performance")
#     plt.title(model)
#     plt.show()

pearsonr(out["m"], out["r"])
    
import seaborn as sns
import matplotlib.pyplot as plt

for model in model_names:
    temp = out[out.model == model]
    
    # Use seaborn's lmplot to plot regression lines for each "layer".
    # Additional scatter_kws is used to adjust the scatter plot appearance, including the alpha value for opacity.
    sns.lmplot(x="m", y="r", hue="layer", data=temp,
               aspect=1.5, height=2, ci=None, palette="viridis",
               scatter_kws={"alpha": 0.3},  # Adjust alpha for opacity
               legend=False)  # Remove the legend
    
    plt.title(model)
    plt.xlabel("Monolingual performance")
    plt.ylabel("Transfer performance")

    # Adjust the figure size as needed
    plt.gcf().set_size_inches(6, 6)
    plt.gcf().dpi=300
    plt.show()

out.to_csv("other/layerwise_mono_multi.csv")

model_formula = 'r ~ m'
mixed_effects_model = smf.mixedlm(model_formula, data=out, groups=out['lang'], 
                                  re_formula='1', vc_formula={'model': '0 + C(model)'})
model_result = mixed_effects_model.fit()
print(model_result.summary())


monol_results = load_monol("bert_base")[6]