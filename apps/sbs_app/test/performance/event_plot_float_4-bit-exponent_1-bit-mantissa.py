import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000, 0.001, 0.053, 0.057, 0.710, 0.714, 0.841, 0.844, 0.966, 0.971, 1.253, 1.256, 1.532, 1.536, 1.570, 1.574, 1.663, 1.667])
latency = np.array([1.672, 0.055, 0.307, 0.655, 1.309, 0.130, 1.098, 0.125, 1.098, 0.284, 1.199, 0.278, 1.184, 0.037, 0.484, 0.091, 0.438, 0.003])
event   = ["SbS_Network", "HX_IN_Software", "HX_IN_Hardware", "H1_CONV_Software", "H1_CONV_Hardware", "H2_POOL_Software", "H2_POOL_Hardware", "H2_POOL_Software", "H2_POOL_Hardware", "H3_CONV_Software", "H3_CONV_Hardware", "H3_CONV_Software", "H3_CONV_Hardware", "H4_POOL_Software", "H4_POOL_Hardware", "H5_DENSE_Software", "H5_DENSE_Hardware", "HY_OUT_Software"]
colors = ["#94c4df", "#4a98c9", "#1864ab", "#4a98c9", "#1864ab", "#4a98c9", "#1864ab", "#4a98c9", "#1864ab", "#4a98c9", "#1864ab", "#4a98c9", "#1864ab", "#4a98c9", "#1864ab", "#4a98c9", "#1864ab", "#4a98c9"]


ax1.barh(range(len(begin)),  latency, left=begin, color=colors)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (mS)")
plt.ylabel("Task")

data = [[ 0.001, 0.057, 0.714, 0.844, 0.971, 1.256, 1.536, 1.574, 1.667],
        [ 0.055, 0.655, 0.130, 0.125, 0.284, 0.278, 0.037, 0.091, 0.003],
        [ 0.307, 1.309, 1.098, 1.098, 1.199, 1.184, 0.484, 0.438, 0.000]]

columns = ("HX_IN", "H1_CONV", "H2_POOL", "H2_POOL", "H3_CONV", "H3_CONV", "H4_POOL", "H5_DENSE", "HY_OUT")
rows = ["Hardware", "Software", "II OFFSET"]

# Get some pastel shades for the colors
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    ax2.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(data[row])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

plt.sca(ax2)
# Add a table at the bottom of the axes
the_table = ax2.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom',
                      fontsize='xx-small')

the_table.auto_set_font_size(False)
the_table.set_fontsize(7)


# Adjust layout to make room for the table:

plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Latency (mS)")

plt.xticks([])
ax2.grid(linestyle = ':')


plt.show()