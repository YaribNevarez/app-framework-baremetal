import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

x = [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200 ]

###############
y1 = [ 0.24, 0.82, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y1, label="Float8 - Noise 0.00")

y2 = [ 0.21, 0.70, 0.87, 0.97, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y2, label="Float8 - Noise 0.50")

y3 = [ 0.15, 0.61, 0.80, 0.88, 0.93, 0.96, 0.97, 0.97, 0.95, 0.97, 0.94 ]
plt.plot(x, y3, label="Float8 - Noise 0.55")

y4 = [ 0.11, 0.35, 0.69, 0.78, 0.80, 0.82, 0.85, 0.83, 0.87, 0.86, 0.84 ]
plt.plot(x, y4, label="Float8 - Noise 0.60")

#y5 = [ 0.11, 0.23, 0.41, 0.52, 0.52, 0.58, 0.64, 0.62, 0.67, 0.64, 0.63 ]
#plt.plot(x, y5, label="Float8 - Noise 0.65")

#y6 = [ 0.05, 0.09, 0.14, 0.22, 0.26, 0.33, 0.32, 0.38, 0.37, 0.37, 0.37 ]
#plt.plot(x, y6, label="Float8 - Noise 0.70")

#y7 = [ 0.05, 0.02, 0.05, 0.07, 0.08, 0.09, 0.08, 0.15, 0.18, 0.14, 0.12 ]
#plt.plot(x, y7, label="Float8 - Noise 0.75")



yb1 = [ 0.16, 0.63, 0.82, 0.96, 0.99, 1.00, 0.98, 0.99, 1.00, 0.99, 0.99 ]
plt.plot(x, yb1, label="Float4 - Noise 0.00")

yb2 = [ 0.11, 0.47, 0.69, 0.79, 0.86, 0.90, 0.91, 0.87, 0.96, 0.94, 0.95 ]
plt.plot(x, yb2, label="Float4 - Noise 0.50")

yb3 = [ 0.07, 0.35, 0.60, 0.72, 0.73, 0.79, 0.83, 0.82, 0.84, 0.85, 0.84 ]
plt.plot(x, yb3, label="Float4 - Noise 0.55")

yb4 = [ 0.11, 0.19, 0.36, 0.43, 0.55, 0.57, 0.57, 0.56, 0.62, 0.64, 0.62 ]
plt.plot(x, yb4, label="Float4 - Noise 0.60")

#yb5 = [ 0.06, 0.06, 0.12, 0.17, 0.26, 0.28, 0.28, 0.33, 0.32, 0.31, 0.30 ]
#plt.plot(x, yb5, label="Float4 - Noise 0.65")

#yb6 = [ 0.05, 0.04, 0.03, 0.10, 0.06, 0.10, 0.07, 0.13, 0.10, 0.11, 0.08 ]
#plt.plot(x, yb6, label="Float4 - Noise 0.70")

#yb7 = [ 0.03, 0.02, 0.02, 0.02, 0.03, 0.05, 0.04, 0.06, 0.02, 0.05, 0.04 ]
#plt.plot(x, yb7, label="Float4 - Noise 0.75")

plt.xlabel('Spikes')
plt.ylabel('Accuracy')

plt.title('Inference accuracy')

plt.legend()
plt.grid()

plt.show()