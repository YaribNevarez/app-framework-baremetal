import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

x = [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200 ]

y1 = [ .18, .75, .93 , 1, 1, 1, 1, 1, 1, 1, 1]
plt.plot(x, y1, label="Simulation. Float8")

###############
y2 = [ 0.14, 0.61, 0.85, 0.95, 0.97, 1.00, 1.00, 1.00, 0.99, 1.00, 0.99 ]
plt.plot(x, y2, label="Simulation. Float4, in $h_\mu(i) W(s_t|i)$, and $\sum_j h_\mu(j) W(s_t|j)$")

###############
#ya = [ 0.28, 0.74, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
#plt.plot(x, ya, label="Simulation. Float8, $\sum_j h_\mu(j) W(s_t|j)$")

###############
#y3 = [ 0.11, 0.19, 0.26, 0.43, 0.65, 0.75, 0.88, 0.93, 0.95, 0.96, 0.95 ]
#plt.plot(x, y3, label="Hardware @ 200MHz. Opt Add, W Float8, $\sum_j h_\mu(j) W(s_t|j)$")

###############
#y4 = [ 0.14, 0.16, 0.28, 0.42, 0.65, 0.78, 0.87, 0.94, 0.95, 0.97, 0.97 ]
#plt.plot(x, y4, label="Hardware @ 120MHz. Opt Add, W Float8, $\sum_j h_\mu(j) W(s_t|j)$")

#y_hw_sw = [ 0.17, 0.36, 0.50, 0.67, 0.88, 0.91, 0.94, 0.97, 0.98, 0.98, 0.99 ]
#plt.plot(x, y_hw_sw, label="Float8 HW+SW")

plt.xlabel('Spikes')
plt.ylabel('Accuracy')

plt.title('Inference accuracy')

plt.legend()
plt.grid()

plt.show()