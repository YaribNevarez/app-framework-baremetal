import matplotlib.pyplot as plt
import numpy as np

begin   = np.array([0.000,0.001,0.001,0.583,0.587,0.587,1.404,1.408,1.408,1.799,1.803,1.803,5.282,5.285,5.286,5.336,5.340,5.340,5.431,5.435,5.435,5.438])
latency = np.array([5.442,0.585,0.585,1.540,0.820,0.819,2.141,0.394,0.393,1.661,3.482,3.481,5.430,0.053,0.053,0.355,0.094,0.094,0.755,0.006,0.005,0.006])
event   = ["SbS Network","HX_INPUT_LAYER","Partition","Hardware","H1_CONVOLUTION_LAYER","Partition","Hardware","H2_POOLING_LAYER","Partition","Hardware","H3_CONVOLUTION_LAYER","Partition","Hardware","H4_POOLING_LAYER","Partition","Hardware","H5_FULLY_CONNECTED_LAYER","Partition","Hardware","HY_OUTPUT_LAYER","Partition","Hardware"]

plt.barh(range(len(begin)),  latency, left=begin)
plt.grid(color = 'g', linestyle = ':')

plt.yticks(range(len(begin)), event)
plt.show()


data = [[ 66386, 174296,  75131, 577908,  32015],
        [ 58230, 381139,  78045,  99308, 160454],
        [ 89135,  80552, 152558, 497981, 603535],
        [ 78415,  81858, 150656, 193263,  69638],
        [139361, 331509, 343164, 781380,  52269]]

columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

values = np.arange(0, 2500, 500)
value_increment = 1000

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % x for x in data[row]])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Loss in ${0}'s".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')

plt.show()