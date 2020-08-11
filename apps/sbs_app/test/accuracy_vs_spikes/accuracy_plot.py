import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

x = [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200 ]

#y1 = [ .18, .75, .93 , 1, 1, 1, 1, 1, 1, 1, 1]
#plt.plot(x, y1, label="Float32 - SW")

###############
y2 = [ 0.10, 0.19, 0.22, 0.32, 0.47, 0.65, 0.74, 0.86, 0.90, 0.92, 0.96 ]
plt.plot(x, y2, label="FPGA. Float4")

y3 = [ 0.12, 0.14, 0.32, 0.45, 0.61, 0.76, 0.85, 0.92, 0.95, 0.95, 0.94 ]
plt.plot(x, y3, label="FPGA. Float5")

y4 = [ 0.15, 0.19, 0.26, 0.40, 0.59, 0.78, 0.85, 0.95, 0.96, 0.96, 0.96 ]
plt.plot(x, y4, label="FPGA. Float6")

y5 = [ 0.12, 0.16, 0.26, 0.38, 0.60, 0.78, 0.85, 0.93, 0.94, 0.96, 0.96 ]
plt.plot(x, y5, label="FPGA. Float7")

#y1 = [ .18, .75, .93 , 1, 1, 1, 1, 1, 1, 1, 1]
#plt.plot(x, y1, label="Simulation. Float8")

#y6 = [ 0.09, 0.17, 0.28, 0.41, 0.62, 0.77, 0.88, 0.94, 0.96, 0.94, 0.97 ]
#plt.plot(x, y6, label="FPGA. Float8")


plt.xlabel('Spikes')
plt.ylabel('Accuracy')

plt.title('Inference accuracy')

plt.legend()
plt.grid()

plt.show()