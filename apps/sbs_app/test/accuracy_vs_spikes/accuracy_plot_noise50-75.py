import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

x = [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200 ]

###############
y1 = [ 0.24, 0.82, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y1, label="4-bit explicit mantissa - 0% Noise")

y2 = [ 0.21, 0.70, 0.87, 0.97, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y2, label="4-bit explicit mantissa - 50% Noise")

y3 = [ 0.15, 0.61, 0.80, 0.88, 0.93, 0.96, 0.97, 0.97, 0.95, 0.97, 0.94 ]
plt.plot(x, y3, label="4-bit explicit mantissa - 55% Noise")

y4 = [ 0.11, 0.35, 0.69, 0.78, 0.80, 0.82, 0.85, 0.83, 0.87, 0.86, 0.84 ]
plt.plot(x, y4, label="4-bit explicit mantissa - 60% Noise")

#y5 = [ 0.11, 0.23, 0.41, 0.52, 0.52, 0.58, 0.64, 0.62, 0.67, 0.64, 0.63 ]
#plt.plot(x, y5, label="Float8 - Noise 0.65")

#y6 = [ 0.05, 0.09, 0.14, 0.22, 0.26, 0.33, 0.32, 0.38, 0.37, 0.37, 0.37 ]
#plt.plot(x, y6, label="Float8 - Noise 0.70")

#y7 = [ 0.05, 0.02, 0.05, 0.07, 0.08, 0.09, 0.08, 0.15, 0.18, 0.14, 0.12 ]
#plt.plot(x, y7, label="Float8 - Noise 0.75")



yb1 = [ 0.20, 0.77, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, yb1, label="1-bit explicit mantissa - 0% Noise")

yb2 = [ 0.22, 0.72, 0.88, 0.96, 0.98, 0.99, 1.00, 1.00, 0.99, 0.99, 1.00 ]
plt.plot(x, yb2, label="1-bit explicit mantissa - 50% Noise")

yb3 = [ 0.19, 0.62, 0.79, 0.85, 0.91, 0.92, 0.93, 0.97, 0.94, 0.96, 0.93 ]
plt.plot(x, yb3, label="1-bit explicit mantissa - 55% Noise")

yb4 = [ 0.16, 0.35, 0.68, 0.77, 0.80, 0.82, 0.84, 0.85, 0.86, 0.86, 0.87 ]
plt.plot(x, yb4, label="1-bit explicit mantissa - 60% Noise")

#yb5 = [ 0.11, 0.21, 0.40, 0.51, 0.53, 0.62, 0.68, 0.64, 0.65, 0.66, 0.62 ]
#plt.plot(x, yb5, label="Float5 - Noise 0.65")

#yb6 = [ 0.06, 0.10, 0.18, 0.27, 0.32, 0.36, 0.40, 0.37, 0.40, 0.33, 0.38 ]
#plt.plot(x, yb6, label="Float5 - Noise 0.70")

#yb7 = [ 0.04, 0.04, 0.04, 0.09, 0.10, 0.10, 0.09, 0.20, 0.21, 0.15, 0.14 ]
#plt.plot(x, yb7, label="Float5 - Noise 0.75")

plt.xlabel('Spikes')
plt.ylabel('Accuracy')

plt.title('Inference Custom Float Quantization (4-bit exponent)')

plt.legend()
plt.grid()

plt.show()