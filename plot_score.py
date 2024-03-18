import numpy as np
import matplotlib.pyplot as plt
import os
ctr = 0
lines = []
lines_sparse = []
file = open('running_scores_avg.txt', 'r')
if os.path.getsize('running_scores_avg.txt'):
# Read all lines from the file and convert them to floats
    for line in file:
        lines.append(float(line.strip()))
        ctr += 1

    plt.plot(range(len(lines)), lines)
    plt.savefig("scores_training_plot.png")