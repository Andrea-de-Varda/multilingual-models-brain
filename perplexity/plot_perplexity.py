import numpy as np
import pandas as pd
import re
from os import chdir
import pickle
import matplotlib.pyplot as plt
import glob
from scipy.stats import pearsonr, sem, norm
from sklearn.linear_model import LinearRegression

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

chdir("/home/dev/Documents/PhD/Alice")

# Specifying model names
model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 5, 5, 5, 5] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]
xglm_langs  = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

ppx = glob.glob("perplexity/*.pkl")

out = []
for filename in ppx:
    with open(filename, 'rb') as file:
        loaded = pickle.load(file)
    mod = re.sub(r"(perplexity\/PERPLEXITY\_results\_)(.*)(.pkl)", r"\2", filename)
    if mod in ["xglm_small", "xglm_med", "xglm_large", "xglm_xl"]:
        for lang, v in loaded.items():
            if lang in xglm_langs:
                out.append([lang, mod, v["result"], v["random"]])
    elif mod == "mgpt":
        for lang, v in loaded.items():
            if lang in mgpt_langs:
                out.append([lang, mod, v["result"], v["random"]])
    else:
        for lang, v in loaded.items():
            if lang in all_codes:
                out.append([lang, mod, v["result"], v["random"]])

out_all = pd.DataFrame(out, columns = ["lang", "mod", "ppx", "ppx_random"])
out_all.replace({'mod':{'mbert':'bert_base'}, 'lang':{'it':'ita'}}, inplace = True)
#out_all["ppx_diff"] = out_all["ppx"] - out_all["ppx_random"]
#out_all["ppx_ratio"] = out_all["ppx"] / out_all["ppx_random"]
# out_all.groupby("mod").agg({"ppx_diff" : "mean", "mod" : "max"})

# residuals
model = LinearRegression()
model.fit(out_all[["ppx_random"]], out_all["ppx"])
out_all["ppx_residuals"] = out_all["ppx"] - model.predict(out_all[["ppx_random"]])
#out_all.to_csv("other/synonyms/perplexity_results.csv", index=False)

print(pearsonr(out_all["ppx_residuals"], out_all["ppx_random"])) # check

ppx_res = out_all.groupby("mod").agg(
    ppx_mean=("ppx_residuals", "mean"),
    ppx_se=("ppx_residuals", sem),
    mod=("mod", "max")
)
ppx_res["Model"] = ppx_res["mod"].map(names_nice_dict)
ppx_res = ppx_res.reset_index(drop=True)
#ppx_res.to_csv("results/perplexity_results.csv", index=False)

# encoding results 
encoding = pd.read_csv("results/mono_multi.csv")

mono_multi = pd.merge(encoding, ppx_res)
del mono_multi["mod"]

################################################################################

############
# PLOTTING #
############

########
# MONO #
########

r, p = pearsonr(mono_multi['Score_mono'], mono_multi['ppx_mean'])

plt.figure(figsize=(7*.9, 9.5*.9), dpi=400)
plt.scatter(mono_multi['ppx_mean'], mono_multi['Score_mono'], color=mono_multi['color'], alpha=1, s = 200)

coefficients = np.polyfit(mono_multi['ppx_mean'], mono_multi['Score_mono'], 1)
polynomial = np.poly1d(coefficients)
x_values = np.linspace(min(mono_multi['ppx_mean'])-8, max(mono_multi['ppx_mean'])+6, 100)
y_values = polynomial(x_values)

plt.plot(x_values, y_values, ls='--', c='gray')

for i in range(len(mono_multi)):
    plt.errorbar(mono_multi['ppx_mean'][i], mono_multi['Score_mono'][i],
                 xerr=mono_multi['ppx_se'][i], yerr=mono_multi['se_mono'][i],
                 fmt='o', color=mono_multi['color'][i], zorder = 5)
annotations = {
    'NLLB$_{d-small}$': (.09,-0.022),
    'NLLB$_{d-large}$': (.07,-.036),
    'NLLB$_{large}$': (-.08,-.048),
    'XLM-Align': (.06,-.02),
    'InfoXLM$_{small}$': (.04,.03),
    'InfoXLM$_{large}$': (-.055,.022),
    'mMiniLM': (-.09,-.075),
    'XLM-R$_{base}$': (-0.08,-0.004),
    'XLM-R$_{large}$': (-.077,.07),
    'DistilmBERT': (-.09,-.054),
    'mBERT': (-.04,-.046),
    'mDeBERTa': (0.09,.012),
    'mT5$_{small}$': (.055,-.04),
    'mT5$_{base}$': (-.06,-0.1),
    'mT5$_{large}$': (-0.14,-0.058),
    'mGPT': (-.055,-0.02),
    'XGLM$_{small}$': (.05,.057),
    'XGLM$_{med}$': (-.05,-.03),
    'XGLM$_{large}$': (.06,.034),
    'XGLM$_{xl}$': (-.05,.013)
}

to_print = ['NLLB$_{d-small}$', 'NLLB$_{large}$', 'InfoXLM$_{large}$', 'mMiniLM', 'XLM-R$_{large}$', 'DistilmBERT', 'mBERT', 'mT5$_{small}$', 'mT5$_{base}$', 'mT5$_{large}$', 'mGPT', 'XGLM$_{small}$', 'XGLM$_{med}$', 'XGLM$_{large}$', 'XGLM$_{xl}$']

#annotations.keys()

models_with_lines = names_formatted

# for i in range(len(mono_multi)):
#     model = mono_multi['Model'][i]
#     if model in to_print:
#         offset_x, offset_y = annotations.get(model, (0.02, 0.02))
#         offset_x = offset_x * 300
#         plt.text(mono_multi['ppx_mean'][i] + offset_x, mono_multi['Score_mono'][i] + offset_y,
#                  model, fontsize=12, ha='center', va='bottom', alpha = 0.3, bbox=dict(facecolor='white', alpha=1, zorder = 4, edgecolor='#D3D3D3'))
#         if model in models_with_lines:
#             plt.plot([mono_multi['ppx_mean'][i], mono_multi['ppx_mean'][i] + offset_x],
#                      [mono_multi['Score_mono'][i], mono_multi['Score_mono'][i] + offset_y],
#                      color='black', alpha = 0.3, lw=1, zorder = 1)
plt.text(0.98, 0.98, f"r = {round(r, 2)}, p = {round(p, 4)}", 
         fontsize=15, ha='right', va='top', alpha=1, 
         bbox=dict(facecolor='white', alpha=0.7), 
         transform=plt.gca().transAxes)

plt.xlabel('Perplexity (residual)', fontsize = 17)
plt.ylabel('R within-languages', fontsize = 17)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(-90, 45)
#plt.ylim(0.15, 0.6)
plt.yticks([0.2, 0.3, 0.4, 0.5])
plt.grid(True)
plt.show()


#########
# MULTI #
#########


r, p = pearsonr(mono_multi['Score_multi'], mono_multi['ppx_mean']) 

plt.figure(figsize=(7*.9, 9.5*.9), dpi=400)
plt.scatter(mono_multi['ppx_mean'], mono_multi['Score_multi'], color=mono_multi['color'], alpha=1, s = 200)

coefficients = np.polyfit(mono_multi['ppx_mean'], mono_multi['Score_multi'], 1)
polynomial = np.poly1d(coefficients)
x_values = np.linspace(min(mono_multi['ppx_mean'])-8, max(mono_multi['ppx_mean'])+6, 100)
y_values = polynomial(x_values)

plt.plot(x_values, y_values, ls='--', c='gray')

for i in range(len(mono_multi)):
    plt.errorbar(mono_multi['ppx_mean'][i], mono_multi['Score_multi'][i],
                 xerr=mono_multi['ppx_se'][i], yerr=mono_multi['se_multi'][i],
                 fmt='o', color=mono_multi['color'][i], zorder = 5)
annotations = {
    'NLLB$_{d-small}$': (.082,-0.01),
    'NLLB$_{d-large}$': (.075,-0.015),
    'NLLB$_{large}$': (-.07,-.014),
    'XLM-Align': (-.107,.033),
    'InfoXLM$_{small}$': (-.04,-.022),
    'InfoXLM$_{large}$': (.08,-.014),
    'mMiniLM': (-.05,-.021),
    'XLM-R$_{base}$': (-0.09,-0.003),
    'XLM-R$_{large}$': (-.06,.028),
    'DistilmBERT': (-.08,-.02),
    'mBERT': (-.042,.005),
    'mDeBERTa': (-0.05,.008),
    'mT5$_{small}$': (.07,-.01),
    'mT5$_{base}$': (.08,-.01),
    'mT5$_{large}$': (-.1,0.0065),
    'mGPT': (-.07,-0.006),
    'XGLM$_{small}$': (-.11,.02),
    'XGLM$_{med}$': (-.07,-0.009),
    'XGLM$_{large}$': (-.06,-0.012),
    'XGLM$_{xl}$': (-.05,.008)
}

to_print = ['NLLB$_{d-small}$', 'NLLB$_{large}$', 'XLM-Align', 'mMiniLM', 'XLM-R$_{base}$', 'XLM-R$_{large}$', 'DistilmBERT', 'mBERT', 'mT5$_{small}$', 'mT5$_{base}$', 'mT5$_{large}$', 'mGPT', 'XGLM$_{small}$', 'XGLM$_{large}$', 'XGLM$_{xl}$']

#annotations.keys()

models_with_lines = names_formatted # CHANGE HERE TO PRINT CORRECTLY

# for i in range(len(mono_multi)):
#     model = mono_multi['Model'][i]
#     if model in to_print:
#         offset_x, offset_y = annotations.get(model, (0.02, 0.02))
#         offset_x = offset_x * 300
#         plt.text(mono_multi['ppx_mean'][i] + offset_x, mono_multi['Score_multi'][i] + offset_y,
#                   model, fontsize=12, ha='center', va='bottom', alpha = 0.3, bbox=dict(facecolor='white', alpha=1, zorder = 4, edgecolor='#D3D3D3'))
#         if model in models_with_lines:
#             plt.plot([mono_multi['ppx_mean'][i], mono_multi['ppx_mean'][i] + offset_x],
#                       [mono_multi['Score_multi'][i], mono_multi['Score_multi'][i] + offset_y],
#                       color='black', alpha = 0.3, lw=1, zorder = 1)
plt.text(0.98, 0.98, f"r = {round(r, 2)}, p = {round(p, 4)}", 
         fontsize=15, ha='right', va='top', alpha=1, 
         bbox=dict(facecolor='white', alpha=0.7), 
         transform=plt.gca().transAxes, clip_on=True)
ax = plt.gca()
margin_low = 0.025
margin_high = 0.045
lo = mono_multi['Score_multi'].min() - margin_low
hi = mono_multi['Score_multi'].max() + margin_high
ax.set_ylim(lo, hi)
plt.xlabel('Perplexity (residual)', fontsize = 17)
plt.ylabel('R across-languages', fontsize = 17)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(-90, 45)
# plt.ylim(0.0, 0.35)
plt.yticks([0.1, 0.2])
plt.grid(True)
plt.show()

# check if corrs are sig different

def r_to_z(r1, r2, n = 130):    
    # fisher r-to-z transformation
    z_1 = np.arctanh(r1)
    z_2 = np.arctanh(r2)
    z_diff = z_1 - z_2
    # standard error of difference
    se_diff = np.sqrt(2*(1/(n-3)))
    z_stat = z_diff / se_diff
    p = 2 * (1 - norm.cdf(np.abs(z_stat))) # two tailed
    return z_stat, p

r_to_z(pearsonr(mono_multi['Score_multi'], mono_multi['ppx_mean'])[0], pearsonr(mono_multi['Score_mono'], mono_multi['ppx_mean'])[0], n = 20)

#####################
# at the lang level #
#####################

out_all = pd.read_csv("other/synonyms/perplexity_results.csv")
out_all.columns
