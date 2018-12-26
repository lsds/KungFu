import numpy as np

import matplotlib.pyplot as plt


machineCountThroughputMap = dict({1 : 215.2, 2 : 357.8, 3 : 538.8, 4 : 653.7})

n, bins, patches = plt.hist(x=machineCountThroughputMap, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Number of 4-GPU Machines')
plt.title('ResNet-50 Scaling with Horovod')
plt.ylabel('Images/sec')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
