import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

x = [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200 ]

y1 = [ .18, .75, .93 , 1, 1, 1, 1, 1, 1, 1, 1]
plt.plot(x, y1, label="Simulation. Float32")

y2 = [ 0.24, 0.82, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 0.99, 1.00 ]
plt.plot(x, y2, label="Simulation. Float8. Custom logic")

###############

y8 = [ 0.20, 0.79, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y8, label="FPGA @ 200MHz. Float8. Custom logic")

yf4 = [ 0.16, 0.63, 0.82, 0.96, 0.99, 1.00, 0.98, 0.99, 1.00, 0.99, 0.99 ]
plt.plot(x, yf4, label="FPGA @ 200MHz. Float4. Custom logic")

yf5 = [ 0.20, 0.77, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, yf5, label="FPGA @ 200MHz. Float5. Custom logic")

###############
#y21 = [ 0.10, 0.19, 0.22, 0.32, 0.47, 0.65, 0.74, 0.86, 0.90, 0.92, 0.96 ]
#plt.plot(x, y21, label="FPGA. Float4")

#y31 = [ 0.12, 0.14, 0.32, 0.45, 0.61, 0.76, 0.85, 0.92, 0.95, 0.95, 0.94 ]
#plt.plot(x, y31, label="FPGA. Float5")

#y41 = [ 0.15, 0.19, 0.26, 0.40, 0.59, 0.78, 0.85, 0.95, 0.96, 0.96, 0.96 ]
#plt.plot(x, y41, label="FPGA. Float6")

#y51 = [ 0.12, 0.16, 0.26, 0.38, 0.60, 0.78, 0.85, 0.93, 0.94, 0.96, 0.96 ]
#plt.plot(x, y51, label="FPGA. Float7")


plt.xlabel('Spikes')
plt.ylabel('Accuracy')

plt.title('Inference accuracy')

plt.legend()
plt.grid()

plt.show()