import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


dades01 = [54,43,24,104,32,63,57,14,32,12]
dades02 = [35,23,14,54,24,33,43,55,23,11]
dades03 = [12,65,24,32,13,54,23,32,12,43]

df_3d = pd.DataFrame([dades01, dades02, dades03]).transpose()
colors = ['r','b','g','y','b','p']


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z= list(df_3d)
for n, i in enumerate(df_3d):
    print 'n',n
    xs = np.arange(len(df_3d[i]))
    ys = [i for i in df_3d[i]]
    zs = z[n]

    cs = colors[n]
    print ' xs:', xs,'ys:', ys, 'zs',zs, ' cs: ',cs
    ax.bar(xs, ys, zs, zdir='y', color=cs, alpha=0.8)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()