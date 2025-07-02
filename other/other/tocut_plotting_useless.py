sns.set_style('whitegrid')
sns.set_context('talk')

# Determine the number of unique models for the hue mapping and set the palette
n_models = out_dfs['Model'].nunique()

# Create a FacetGrid to make a grid of plots
g = sns.FacetGrid(out_dfs, col='Class', col_wrap=2, height=6, aspect=1.33)

# We will keep track of the ylims here
all_ylims = []

# Map the lineplot to each subset of the data
for cls in out_dfs['Class'].unique():
    class_data = out_dfs[out_dfs['Class'] == cls]
    # Extract the min and max metric value for the current class
    class_ylims = (class_data['m'].min(), class_data['m'].max())
    all_ylims.append(class_ylims)
    
# Determine global y-limits
global_ylims = (min(all_ylims)[0], max(all_ylims)[1])

# Generate a color for each model
palette = sns.color_palette("tab10", n_colors=n_models)

# Map the lineplot for each class/model combination
g = g.map_dataframe(
    sns.lineplot, 
    x='l', 
    y='m', 
    hue='Model', 
    style='Model', 
    markers=True, 
    dashes=False, 
    palette=palette
)

# Adjust the axis labels, titles, and legend
g.set_axis_labels('Layer position', 'R', fontsize=22, labelpad=15)
g.set_titles('{col_name}', fontsize=26, weight='bold', pad=20)

# Set the same y-limits for all plots
for ax in g.axes.flatten():
    ax.set_ylim(global_ylims)

# Customizing the legend
# Get the legend data from the FacetGrid
legend_data = g._legend_data
# Remove the existing legend, since we will draw a new one
g._legend.remove()

# Place the legend outside the plot area
g.fig.subplots_adjust(right=0.85)  # Adjust subplot params for new legend
legend = g.fig.legend(
    handles=legend_data.values(), 
    labels=legend_data.keys(), 
    title='Model', 
    loc='center right', 
    bbox_to_anchor=(1.25, 0.5)
)

plt.setp(legend.get_title(), fontsize='22')  # Title font size
plt.setp(legend.get_texts(), fontsize='18')  # Label font size

# Show plot
plt.show()



# lineplot with single languages for best model

# monolingual
colname = "m"
results_dict = load("bert_base") # do with both xglm_xl and mBERT
langs = results_dict[0]["lang"].tolist()
layers = max(results_dict.keys())+1

res_lang = []
for lang in langs:
    for layer in range(layers):
        res_l = results_dict[layer]
        res_l_lang = res_l[res_l["lang"] == lang]
        res = res_l_lang[colname].values[0]
        res_lang.append([layer, res, lang])
res_lang = pd.DataFrame(res_lang, columns = ["Layer", "R", "Language"])

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(14*.7, 12*.7), dpi = 300)
n_classes = res_lang['Language'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)
ax = sns.lineplot(
    data=res_lang,
    x='Layer', 
    y='R', 
    hue='Language',  # Color by class
    style='Language',  # Different markers for each name
    markers=True, 
    dashes=False, 
    palette=palette
)
ax.set_xlabel('Layer', fontsize=27, labelpad=15)
ax.set_ylabel('R', fontsize=27, labelpad=15)
ax.set_title('Layerwise results by language', fontsize=30, weight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(title_fontsize='22', fontsize='18', loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
frame = legend.get_frame()
frame.set_color('white')
frame.set_edgecolor('black')
ax.set_xlim([res_lang["Layer"].min()-.5, res_lang['Layer'].max()+.5])
ax.set_ylim([res_lang['R'].min() - 0.05, res_lang['R'].max() + 0.05])
plt.show()

# transfer
colname = "r"
results_dict = load("xglm_xl", monol=False) # do with both xglm_xl and mBERT
langs = results_dict[0]["lang"].tolist()
layers = max(results_dict.keys())+1

res_lang = []
for lang in langs:
    for layer in range(layers):
        res_l = results_dict[layer]
        res_l_lang = res_l[res_l["lang"] == lang]
        res = res_l_lang[colname].values[0]
        res_lang.append([layer, res, lang])
res_lang = pd.DataFrame(res_lang, columns = ["Layer", "R", "Language"])

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(14*.7, 12*.7), dpi = 300)
n_classes = res_lang['Language'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)
ax = sns.lineplot(
    data=res_lang,
    x='Layer', 
    y='R', 
    hue='Language',  # Color by class
    style='Language',  # Different markers for each name
    markers=True, 
    dashes=False, 
    palette=palette
)
ax.set_xlabel('Layer', fontsize=27, labelpad=15)
ax.set_ylabel('R', fontsize=27, labelpad=15)
ax.set_title('Layerwise results by language', fontsize=30, weight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(title_fontsize='22', fontsize='18', loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
frame = legend.get_frame()
frame.set_color('white')
frame.set_edgecolor('black')
ax.set_xlim([res_lang["Layer"].min()-.5, res_lang['Layer'].max()+.5])
ax.set_ylim([res_lang['R'].min() - 0.05, res_lang['R'].max() + 0.05])
plt.show()








colname = "m"
r_lang = {lang : [] for lang in langs}
sd_lang = {lang : [] for lang in langs}
for model in model_names:
    res_dict = load(model)
    # first selecting best layer
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    df = res_dict[idx_max]
    for lang in langs:
        try:
            therow = df[df.lang == lang]
            r = therow.values[0][1]
            sd = therow.values[0][2]
            r_lang[lang].append(r)
            sd_lang[lang].append(sd)
        except IndexError: # xglm models miss some languages
            r_lang[lang].append(0)
            sd_lang[lang].append(0)

data = pd.DataFrame(r_lang)
data1 = pd.DataFrame(sd_lang) # IMPORTANT: here the SD is the variation across languages

fig, axes = plt.subplots(11, 1, figsize=(10*.9, 15*.9))
yticks = [0, .5, 1]
for i, ax in enumerate(axes):
    ax.bar(data.columns, data.iloc[i], color='indianred', yerr=data1.iloc[i])
    ax.set_ylim(-.1, 1)
    ax.set_yticks(yticks)  # Set the y-ticks to [0, 0.3, 0.6, 1]
    # grid
    ax.xaxis.grid(False)
    ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
    if i < len(axes) - 1:
        ax.set_xticklabels([])  # Hide x-tick labels
    else:
        ax.set_xticklabels([lang_dict[l] for l in data.columns], rotation=45, ha="right")
plt.suptitle('Encoding results by language', y=.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.show()



xglm = load("xglm_xl")
xglm[0].sort_values(by="m", ascending = False)["lang"].tolist()

inv_lang_dict = {v : k for k, v in lang_dict.items()}

langs_nice = [lang_dict[lang] for lang in langs]
print(langs)












