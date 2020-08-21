import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

x = [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200 ]

y1 = [ .18, .75, .93 , 1, 1, 1, 1, 1, 1, 1, 1]
plt.plot(x, y1, label="Simulation. Float32")

###############
y2 = [ 0.24, 0.82, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 0.99, 1.00 ]
plt.plot(x, y2, label="Simulation. Float8. Custom SHIFT-ADD-MUL")


plt.xlabel('Spikes')
plt.ylabel('Accuracy')

plt.title('Inference accuracy')

plt.legend()
plt.grid()

plt.show()