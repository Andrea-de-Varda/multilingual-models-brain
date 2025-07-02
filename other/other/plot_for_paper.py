def plot_aggregate(df, title, ylim = None, ylimstart = None):
    plt.figure(figsize=(14*.7, 11.5*.7), dpi = 300)  
    sns.set_context("talk")
    palette = sns.color_palette("tab10")  # Colorblind-friendly palette
    ax = sns.barplot(x='Model', y='Score', hue='Family', data=df,
                     dodge=False, palette=palette, edgecolor='.2')
    for i in range(len(df['Score'])):
        plt.errorbar(i, df['Score'][i], yerr=df['sd'][i], fmt='none', capsize=5, ecolor='black', capthick=2)
    plt.title(title, fontsize=30, weight='bold', pad=20)
    plt.xlabel('Model', fontsize=27, labelpad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    plt.ylim(ylimstart, ylim)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=23)
    #leg = plt.legend(title='Model Family', ncols = 4, title_fontsize='20', fontsize='18', loc='upper left', bbox_to_anchor=(1, 1))
    #for legobj in leg.legendHandles:
    #    legobj.set_linewidth(4.0)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    ax.get_legend().remove()
    plt.show()

plot_aggregate(median_layer, "",  ylim = .85, ylimstart = -.2)
