import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000, 0.001, 1.170, 6.000, 9.970, 31.702, 32.572, 36.024])
latency = np.array([36.029, 1.167, 4.829, 3.967, 21.731, 0.868, 3.450, 0.004])
event   = ["SbS_Network", "HX_IN_Software", "H1_CONV_Software", "H2_POOL_Software", "H3_CONV_Software", "H4_POOL_Software", "H5_DENSE_Software", "HY_OUT_Software"]

ax1.barh(range(len(begin)),  latency, left=begin)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (mS)")
plt.ylabel("Task")

data = [[ 0.001, 1.170, 6.000, 9.970, 31.702, 32.572, 36.024],
        [ 1.167, 4.829, 3.967, 21.731, 0.868, 3.450, 0.004],
        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]]

columns = ("HX_IN", "H1_CONV", "H2_POOL", "H3_CONV", "H4_POOL", "H5_DENSE", "HY_OUT")
rows = ["Hardware", "Software", "II OFFSET"]

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