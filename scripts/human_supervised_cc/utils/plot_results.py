import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os

results_dir = "/home/scazlab/catkin_ws/src/recycling_stretch/scripts/contrastive_clustering/results"
labels_files = []
for file in os.listdir(results_dir):
    json_file = json.load(open(os.path.join(results_dir, file)))
    labels_files.append(file)
    print(json_file)


labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

# plt.show()