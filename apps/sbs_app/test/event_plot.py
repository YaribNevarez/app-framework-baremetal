import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000, 0.001, 0.546, 0.551, 1.375, 1.380, 1.809, 1.814, 2.687, 2.691, 3.480, 3.484, 3.625, 3.629, 3.633])
latency = np.array([3.637, 0.548, 1.541, 0.827, 2.141, 0.432, 1.666, 0.876, 5.430, 0.791, 0.364, 0.143, 0.755, 0.006, 0.005])
event   = ["SbS Network", "HX_IN_Partition", "HX_IN_Hardware", "H1_CONV_Partition", "H1_CONV_Hardware", "H2_POOL_Partition", "H2_POOL_Hardware", "H3_CONV_Partition", "H3_CONV_Hardware", "H4_POOL_Partition", "H4_POOL_Hardware", "H5_DENSE_Partition", "H5_DENSE_Hardware", "HY_OUT_Partition", "HY_OUT_Hardware"]

ax1.barh(range(len(begin)),  latency, left=begin)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (mS)")
plt.ylabel("Task")

data = [[ 0.001, 0.586, 1.404, 1.803, 2.717, 3.474, 3.625],
        [ 0.583, 0.816, 0.398, 0.912, 0.756, 0.149, 0.007],
        [ 1.540, 2.184, 1.667, 5.430, 0.364, 0.755, 0.005]]

columns = ("HX_IN","H1_CONV","H2_POOL","H3_CONV","H4_POOL","H5_DENSE","HY_OUT")
rows = ["HW", "CPU", "II OFFSET"]

# Get some pastel shades for the colors
colors = plt.cm.Blues(np.linspace(0.5, .75, len(rows)))
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