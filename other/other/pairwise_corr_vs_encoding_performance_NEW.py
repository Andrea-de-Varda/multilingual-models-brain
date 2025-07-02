##########################################################
# plot encoding performance against pairwise correlation #
##########################################################

corr_pairs = pd.read_csv("results/correlations/correlation_participants_new.csv")
corr_pairs = {row["lang"] : row["r"] for index, row in corr_pairs.iterrows()}

model_names_bidirect = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base"]
model_names_xglm = ["xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

colname = "m"
r_lang = {lang : [] for lang in langs}
for model in model_names_bidirect:
    with open(f"results/monolingual_sequential_{model}", 'rb') as handle:
        res_dict = pickle.load(handle)
    # first selecting best layer
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    df = res_dict[idx_max]
    for lang in langs:
        try:
            therow = df[df.lang == lang]
            r = therow.values[0][1]
            r_lang[lang].append(r)
        except IndexError: # xglm models miss some languages
            r_lang[lang].append(np.nan)

r_lang = [[lang_code_dict[k], np.nanmean(v), corr_pairs[lang_code_dict[k]]] for k, v in r_lang.items()]
r_lang = pd.DataFrame(r_lang, columns = ["lang", "score", "corr_pair"])

filt = np.array(~np.isnan(r_lang["score"]))
pearsonr(r_lang["score"][filt], r_lang["corr_pair"][filt])


plt.figure(figsize=(23/3.5, 14/3.5), dpi = 300)
scatter = sns.scatterplot(data=r_lang, 
                          x="corr_pair", 
                          y="score", 
                          #hue="modelkind", 
                          #style="modelkind", 
                          palette="deep",
                          s=100)
sns.regplot(data=r_lang, 
            x="corr_pair", 
            y="score", 
            scatter=False,
            color="gray")
texts = []
for line in range(0, r_lang.shape[0]):
    texts.append(scatter.text(r_lang.corr_pair[line], 
                              r_lang.score[line], 
                              r_lang.lang[line], 
                              horizontalalignment='left', 
                              size='medium', 
                              color='black'))
#adjust_text(texts)
plt.xlabel("Time-series correlation", fontsize=12.2)
plt.ylabel("Encoding performance", fontsize=12.2)
plt.title("Bidirectional models", fontsize=17, weight="bold")
#plt.legend(title="Model Kind", loc="upper left")
#plt.ylim((0, 30))
plt.xlim((0.2, .54))
plt.tick_params(axis='both', labelsize=12)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Model Kind")
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()