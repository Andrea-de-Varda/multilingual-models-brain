best_mono = [get_best_layerwise(load(model, monol = True), colname="m") for model in model_names]
best_mono_sd = [get_best_layerwise(load(model, monol = True), colname="m", give_mean = False) for model in model_names]

best_layer_mono = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_mono,
    'sd' : best_mono_sd
})

mono_multi = pd.merge(best_layer_mono, best_layer_multi, on = "Model", suffixes = ["_mono", "_multi"])
mono_multi["se_mono"] = mono_multi["sd_mono"] / mono_multi["n"]
mono_multi["se_multi"] = mono_multi["sd_multi"] / mono_multi["n"]
mono_multi["color"] = mono_multi["Family"].map(palette_d)

r, p = pearsonr(mono_multi['Score_mono'], mono_multi['Score_multi']) # 0.6293935246947423 0.0029449601326826487

################################################################################

plt.figure(figsize=(7, 9.5), dpi=400)
plt.scatter(mono_multi['Score_mono'], mono_multi['Score_multi'], color=mono_multi['color'], alpha=1, s = 200)

coefficients = np.polyfit(mono_multi['Score_mono'], mono_multi['Score_multi'], 1)
polynomial = np.poly1d(coefficients)
x_values = np.linspace(min(mono_multi['Score_mono']), max(mono_multi['Score_mono']), 100)
y_values = polynomial(x_values)

plt.plot(x_values, y_values, ls='--', c='gray')

for i in range(len(mono_multi)):
    plt.errorbar(mono_multi['Score_mono'][i], mono_multi['Score_multi'][i],
                 xerr=mono_multi['se_mono'][i], yerr=mono_multi['se_multi'][i],
                 fmt='o', color=mono_multi['color'][i], zorder = 5)
annotations = {
    'NLLB$_{d-small}$': (.053,0.007),
    'NLLB$_{d-large}$': (-.031,-.024),
    'NLLB$_{large}$': (.05,-.02),
    'XLM-Align': (-.04,.02),
    'InfoXLM$_{small}$': (-.09,.02),
    'InfoXLM$_{large}$': (-.01,.04),
    'mMiniLM': (-.05,.018),
    'XLM-R$_{base}$': (0.03,0.01),
    'XLM-R$_{large}$': (-.06,.015),
    'DistilmBERT': (0.08,-.024),
    'mBERT': (-.05,.04),
    'mDeBERTa': (0.09,-.015),
    'mT5$_{small}$': (-.03,.015),
    'mT5$_{base}$': (.035,-0.035),
    'mT5$_{large}$': (0.137,0.03),
    'mGPT': (.07,0.02),
    'XGLM$_{small}$': (.05,.01),
    'XGLM$_{med}$': (.05,-.03),
    'XGLM$_{large}$': (.045,-.02),
    'XGLM$_{xl}$': (.04,-.02)
}

models_with_lines = names_formatted

for i in range(len(mono_multi)):
    model = mono_multi['Model'][i]
    offset_x, offset_y = annotations.get(model, (0.02, 0.02))
    
    plt.text(mono_multi['Score_mono'][i] + offset_x, mono_multi['Score_multi'][i] + offset_y,
             model, fontsize=12, ha='center', va='bottom', alpha = 0.3, bbox=dict(facecolor='white', alpha=1, zorder = 4, edgecolor='#D3D3D3'))
    if model in models_with_lines:
        plt.plot([mono_multi['Score_mono'][i], mono_multi['Score_mono'][i] + offset_x],
                 [mono_multi['Score_multi'][i], mono_multi['Score_multi'][i] + offset_y],
                 color='black', alpha = 0.3, lw=1, zorder = 1)
plt.text(0.438, 0.394, f"r = {round(r, 2)}, p = {round(p, 4)}", fontsize=18, ha='center', va='bottom', alpha = 1, bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('R within-language', fontsize = 17)
plt.ylabel('R across-languages', fontsize = 17, rotation=-90, ha='center', labelpad = 26)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(0.14, 0.575)
plt.ylim(None, 0.405)
plt.grid(True)
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.show()
