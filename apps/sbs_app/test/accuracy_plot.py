import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

x = [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200 ]

y1 = [ .18, .75, .93 , 1, 1, 1, 1, 1, 1, 1, 1]
plt.plot(x, y1, label="Simulation. No Opt, Weight Float8")

###############
y2 = [ 0.14, 0.61, 0.85, 0.95, 0.97, 1.00, 1.00, 1.00, 0.99, 1.00, 0.99 ]
plt.plot(x, y2, label="Simulation. Opt Mul, W Float4, ($h_\mu(i) W(s_t|i)$)")

###############
y2 = [ 0.28, 0.74, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 ]
plt.plot(x, y2, label="Simulation. Opt Add, W Float8, $\sum_j h_\mu(j) W(s_t|j)$")

###############
y3 = [ 0.11, 0.19, 0.26, 0.43, 0.65, 0.75, 0.88, 0.93, 0.95, 0.96, 0.95 ]
plt.plot(x, y2, label="Hardware @ 200MHz. Opt Add, W Float8, $\sum_j h_\mu(j) W(s_t|j)$")

###############
y4 = [ 0.14, 0.16, 0.28, 0.42, 0.65, 0.78, 0.87, 0.94, 0.95, 0.97, 0.97 ]
plt.plot(x, y2, label="Hardware @ 120MHz. Opt Add, W Float8, $\sum_j h_\mu(j) W(s_t|j)$")

plt.xlabel('Spikes')
plt.ylabel('Accuracy')

plt.title('Inference')

plt.legend()
plt.grid()

plt.show()