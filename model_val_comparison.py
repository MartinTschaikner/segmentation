import numpy as np
from tkinter.filedialog import askdirectory
from os import walk, path
import matplotlib.pyplot as plt
from itertools import cycle

stack = np.load(r'data\validation_Spectralis_ring_scan\ring_scan_stack.npy')
stack_labels = np.load(r'data\validation_Spectralis_ring_scan\ring_scan_label_stack.npy')

dir_models = askdirectory(title='Choose directory of all validation data to compare!')

model_dirs = [x[0] for x in walk(dir_models)][1:]

all_average_results = []

for i in range(len(model_dirs)):

    scores = np.load(model_dirs[i] + '/scores.npy')
    errors = np.load(model_dirs[i] + '/errors.npy')
    segmentation = np.load(model_dirs[i] + '/segmentation.npy')

    av_scores = np.average(np.average(scores, axis=0), axis=1)
    av_errors = np.nanmean(errors, axis=2)

    av_results = [path.basename(model_dirs[i]), av_scores, av_errors]
    all_average_results.append(av_results)

for i in range(len(model_dirs)):
    print(all_average_results[i][0], '\t', all_average_results[i][1], '\t', all_average_results[i][2])

scores = ['dice', 'accuracy', 'precision', 'specicifity', 'sensitivity', 'balanced acc', 'jaccard', 'tversky7']
layers = ['ILM', 'RNFL', 'RPE']

# plot only dice, jaccard, tversky7
plot_scores_indices = [0, 6, 7]
scores = [scores[i] for i in plot_scores_indices]
for i in range(len(model_dirs)):
    all_average_results[i][1] = all_average_results[i][1][np.array(plot_scores_indices)]

fig, ax = plt.subplots(ncols=1)
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)

for i in range(len(model_dirs)):
    ax.plot(scores, all_average_results[i][1], next(linecycler), label=all_average_results[i][0])

ax.legend()
plt.show()

fig, ax = plt.subplots(ncols=2)

for i in range(len(model_dirs)):
    ax[0].plot(layers, all_average_results[i][2][:, 0], next(linecycler), label=all_average_results[i][0])
    ax[1].plot(layers, all_average_results[i][2][:, 1], next(linecycler), label=all_average_results[i][0])

ax[0].legend()
ax[1].legend()
plt.show()
