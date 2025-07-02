import matplotlib.pyplot as plt

# Define the legend items and their colors
train_names_nice = ["Study I data", "NatStories data"]
colors = ["darkslateblue", "lightsteelblue"]

# Create an empty figure
fig, ax = plt.subplots(figsize=(2, 1), dpi=300)  # Adjust size to fit the legend nicely
ax.axis("off")  # Turn off the axis

# Create legend items
legend_handles = [
    plt.Line2D([0], [0], color=color, lw=10) for color in colors
]

# Add the legend
legend = ax.legend(
    legend_handles,
    train_names_nice,
    loc="center",  # Keep the box centered
    ncol=1,  # Single column for the legend
    frameon=False,  # Add a border around the legend
    fontsize=12,
    title="Encoding models trained on",
    title_fontsize=12,  # Title font size
    handletextpad=1,  # Space between the color patch and text
    borderpad=0,  # Padding between the legend content and the box
    labelspacing=0.3  # Adjust vertical spacing between items
)

# Align everything in the legend box to the left
legend._legend_box.align = "left"
legend.get_title().set_position((-25, 0))  # Move the title slightly to the left

# Ensure proper layout
plt.show()






# Define the legend items and their colors
train_names_nice = ["Tuckute2024 data", "Pereira2018 data"]
colors = ["tab:red", "lightsalmon"]

# Create an empty figure
fig, ax = plt.subplots(figsize=(2, 1), dpi=300)  # Adjust size to fit the legend nicely
ax.axis("off")  # Turn off the axis

# Create legend items
legend_handles = [
    plt.Line2D([0], [0], color=color, lw=10) for color in colors
]

# Add the legend
legend = ax.legend(
    legend_handles,
    train_names_nice,
    loc="center",  # Keep the box centered
    ncol=1,  # Single column for the legend
    frameon=False,  # Add a border around the legend
    fontsize=12,
    title="Encoding models trained on",
    title_fontsize=12,  # Title font size
    handletextpad=1,  # Space between the color patch and text
    borderpad=0,  # Padding between the legend content and the box
    labelspacing=0.3  # Adjust vertical spacing between items
)

# Align everything in the legend box to the left
legend._legend_box.align = "left"
legend.get_title().set_position((-25, 0))  # Move the title slightly to the left

# Ensure proper layout
plt.show()