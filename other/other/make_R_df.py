import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from tqdm import tqdm
from scipy.stats import pearsonr, norm
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import copy
import itertools
import glob
from researchpy import corr_pair

chdir("/home/dev/Documents/PhD/Alice")

all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}

perplex = pd.read_csv("results/perplexity_results.csv")
transf_lin = pd.read_csv("results/transfer_and_lin_features.csv")
del transf_lin['Unnamed: 0']
del perplex['Unnamed: 0']

perplex["lang_mod"] = perplex["lang"].map(lang_code_dict) + "_" + perplex["mod"]
ppx_dict = {row["lang_mod"] : row["ppx"] for index, row in perplex.iterrows()}
transf_lin["lang_mod1"] = transf_lin["Lang1"] + "_" + transf_lin["model"]
transf_lin["lang_mod2"] = transf_lin["Lang2"] + "_" + transf_lin["model"]
transf_lin["ppx_1"] = transf_lin["lang_mod1"].map(ppx_dict)
transf_lin["ppx_2"] = transf_lin["lang_mod2"].map(ppx_dict)

transf_lin.to_csv("results/transfer_analysis_R.csv", index = False)
