import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

x = [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200 ]

#y1 = [ .18, .75, .93 , 1, 1, 1, 1, 1, 1, 1, 1]
#plt.plot(x, y1, label="Float32 - SW")

###############
y2 = [ 0.24, 0.82, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y2, label="FPGA. Float8. Noise 0%")

y3 = [ 0.23, 0.78, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y3, label="FPGA. Float8. Noise 25%")

y4 = [ 0.21, 0.72, 0.91, 0.97, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y4, label="FPGA. Float8. Noise 50%")

y5 = [ 0.04, 0.04, 0.07, 0.05, 0.11, 0.11, 0.12, 0.12, 0.12, 0.11, 0.17 ]
plt.plot(x, y5, label="FPGA. Float8. Noise 75%")

y6 = [ 0.04, 0.02, 0.02, 0.02, 0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02 ]
plt.plot(x, y6, label="FPGA. Float8. Noise 100%")


yb2 = [ 0.17, 0.60, 0.80, 0.95, 1.00, 1.00, 0.98, 0.99, 1.00, 0.99, 1.00 ]
plt.plot(x, yb2, label="FPGA. Float4. Noise 0%")

yb3 = [ 0.11, 0.57, 0.78, 0.93, 0.98, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, yb3, label="FPGA. Float4. Noise 25%")

yb4 = [ 0.12, 0.48, 0.72, 0.83, 0.84, 0.91, 0.92, 0.97, 0.94, 0.95, 0.92 ]
plt.plot(x, yb4, label="FPGA. Float4. Noise 50%")

yb5 = [ 0.01, 0.02, 0.02, 0.04, 0.04, 0.05, 0.04, 0.03, 0.05, 0.04, 0.07 ]
plt.plot(x, yb5, label="FPGA. Float4. Noise 75%")

yb6 = [ 0.04, 0.02, 0.02, 0.02, 0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02 ]
plt.plot(x, yb6, label="FPGA. Float4. Noise 100%")

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