import matplotlib.pyplot as plt
import csv

accelerators = 8
layers = 7
x = [[0] * layers for i in range(layers + accelerators)]
y = [[0] * layers for i in range(layers + accelerators)]

file_name = 'accelerator_activity.csv' 

with open(file_name, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
      for n in range(0, layers):
        x[n].append(float(row[0 + 2*n]))
        y[n].append(float(row[1 + 2*n]) + 2*(n))
      for n in range(layers, layers + accelerators):
        x[n].append(float(row[0 + 2*n]))
        y[n].append(float(row[1 + 2*n]) + 2*(n - layers))

for n in range(0, layers):
  plt.plot(x[n], y[n], label='Layer ' + str(n))

for n in range(0, accelerators):
  plt.plot(x[layers + n], y[layers + n], label='Accelerator ' + str(n))


plt.xlabel('Time(S)')
plt.ylabel('Activity')

plt.title('Accelerator platform')
plt.legend()
plt.show()