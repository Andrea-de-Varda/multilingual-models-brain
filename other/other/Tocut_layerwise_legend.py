fig = plt.figure(figsize=(13*.8, 7*.8), dpi=400)  # Main figure size

# Create a 3x3 grid
gs = gridspec.GridSpec(3, 3, figure=fig)

# Loop through the class order and create subplots within each main grid cell
for i, model_class in enumerate(class_order):
    # Find row and column index
    row = i // 3
    col = i % 3
    
    # Create a nested GridSpec for two subplots
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row, col], wspace=0.1)

    # Plot for out_dfs
    ax1 = fig.add_subplot(inner_gs[0, 1])
    group_data = out_dfs[out_dfs['Class'] == model_class]
    for model in group_data['Model'].unique():
        model_data = group_data[group_data['Model'] == model]
        sns.lineplot(
            data=model_data,
            x='l', 
            y='m',
            color=model_data['color'].iloc[0],
            marker=model_data['marker'].iloc[0],
            label=model,
            ax=ax1, markersize=7
        )
    ax1.set_xticks([.2, .6])
    #ax1.tick_params(axis='x', labelsize=18)
    if row == 2:
        ax1.set_xticks([.25, .5, .75])
        ax1.tick_params(axis='x', labelsize=0)
    else:
        ax1.set_xticks([.25, .5, .75])
        ax1.tick_params(axis='x', labelsize=0)
    #ax1.tick_params(axis='y', labelsize=18)
    ax1.set_xlim([group_data['l'].min()-.02, group_data['l'].max()+.02])
    ax1.set_ylim(-.05, .53)
    ax1.set_ylabel('')  # Remove y-axis label
    ax1.set_xlabel('')  # Remove x-axis label
    ax1.set_yticks([0, 0.2, 0.4])  # Set y-ticks
    ax1.get_legend().remove()
    ax1.spines['top'].set_color('black')
    ax1.spines['top'].set_linewidth(1.5)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_color('black')
    ax1.spines['right'].set_linewidth(1.5)

    # Plot for out_dfs_mono
    ax2 = fig.add_subplot(inner_gs[0, 0])
    group_data_mono = out_dfs_mono[out_dfs_mono['Class'] == model_class]
    for model in group_data_mono['Model'].unique():
        model_data_mono = group_data_mono[group_data_mono['Model'] == model]
        sns.lineplot(
            data=model_data_mono,
            x='l', 
            y='m',
            color=model_data_mono['color'].iloc[0],
            marker=model_data_mono['marker'].iloc[0],
            label=model,
            ax=ax2, markersize=7
        )
    #ax2.set_xticks([.2, .4, .6, .8])
    if row == 2:
        ax2.set_xticks([.25, .5, .75])
        ax2.tick_params(axis='x', labelsize=0)
    else:
        ax2.set_xticks([.25, .5, .75])
        ax2.tick_params(axis='x', labelsize=0)
    if col == 0:
        ax2.tick_params(axis='y', labelsize=16)
    else:
        ax2.set_yticklabels([])
    #ax2.tick_params(axis='x', labelsize=18)
    #ax2.tick_params(axis='y', labelsize=0)
    ax2.set_xlim([group_data_mono['l'].min()-.02, group_data_mono['l'].max()+.02])
    ax2.set_ylim(-.05, .53)
    ax2.set_ylabel('')  # Remove y-axis label
    ax2.set_xlabel('')  # Remove x-axis label
    ax2.set_yticks([0, 0.2, 0.4])  # Set y-ticks
    ax2.get_legend().remove()
    ax2.spines['top'].set_color('black')
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(1.5)

    # Set a common title for the sub-subplots
    fig.text(x=(col / 3) + 0.2 - col*0.02, y=(1 - row / 3) -row*-0.042 -.1, s=model_class, ha='center', fontsize=22)

# Adjust the layout of the figure
fig.text(0.53, 0.02, 'Layer position', ha='center', va='center', fontsize=23)
fig.text(0.00, 0.5, 'R', ha='center', va='center', rotation='vertical', fontsize=23)

# Create a legend with 4 rows and 5 columns, positioned on top
legend_handles = [Line2D([0], [0], color=palette_d[class_dict_nice[model]], 
                         marker=model_markers[model], label=model, linestyle='-', markersize=8) 
                  for model in out_dfs['Model'].unique()]

fig.legend(
    handles=legend_handles, 
    loc='upper center',  # Position the legend on top
    bbox_to_anchor=(0.515, 1.3),  # Centered horizontally above the plot
    fontsize=14.2, 
    title_fontsize=20, 
    title='Model',
    ncol=5,  # 5 columns
    frameon=True,  # Optional: remove legend border
)

fig.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout to make space for the legend

plt.show()





# Create a blank figure
fig = plt.figure(dpi=300, figsize=(8, 2))  # Adjust figsize to control the size of the legend

# Create legend handles
legend_handles = [Line2D([0], [0], color=palette_d[class_dict_nice[model]], 
                         marker=model_markers[model], label=model, linestyle='-', markersize=8) 
                  for model in out_dfs['Model'].unique()]

# Add the legend to the figure
fig.legend(
    handles=legend_handles, 
    loc='center',  # Center the legend in the figure
    fontsize=14.2, 
    title_fontsize=20, 
    ncol=3,  # 5 columns
    frameon=True,  # Show the legend border
)

# Hide axes
plt.axis('off')

plt.show()
