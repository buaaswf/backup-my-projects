import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

min_val, max_val = 0, 15

intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))

ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

for i in xrange(15):
    for j in xrange(15):
        c = intersection_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')