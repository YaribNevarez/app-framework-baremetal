import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000,0.001,0.001,0.595,0.599,0.600,1.424,1.428,1.428,1.813,1.817,1.818,2.733,2.737,2.737,3.484,3.488,3.488,3.637,3.640,3.641,3.645])
latency = np.array([3.649,0.597,0.597,1.539,0.827,0.826,2.175,0.389,0.388,1.666,0.919,0.918,5.430,0.750,0.749,0.364,0.152,0.151,0.755,0.008,0.007,0.005])
event   = ["SbS Network","HX_IN","Partition","Hardware","H1_CONV","Partition","Hardware","H2_POOL","Partition","Hardware","H3_CONV","Partition","Hardware","H4_POOL","Partition","Hardware","H5_DENSE","Partition","Hardware","HY_OUT","Partition","Hardware"]

ax1.barh(range(len(begin)),  latency, left=begin)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (mS)")
plt.ylabel("Task")

data = [[ 0.597,0.826,0.388,0.918,0.749,0.151,0.007],
        [ 1.539,2.175,1.666,5.430,0.364,0.755,0.005]]

columns = ("HX_IN","H1_CONV","H2_POOL","H3_CONV","H4_POOL","H5_DENSE","HY_OUT")
rows = ["HW", "CPU"]

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0.5, .75, len(rows)))
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