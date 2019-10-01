from viewer_validation import plot_validation
import numpy as np
from tkinter.filedialog import askdirectory
import os.path

stack = np.load(r'data\validation_Spectralis_ring_scan\ring_scan_stack.npy')
stack_labels = np.load(r'data\validation_Spectralis_ring_scan\ring_scan_label_stack.npy')

dir_model = askdirectory(title='Choose directory of model data for validation!')
model_name = os.path.basename(dir_model)

scores = np.load(dir_model + '/scores.npy')
errors = np.load(dir_model + '/errors.npy')
segmentation = np.load(dir_model + '/segmentation.npy')

average_scores = np.average(scores, axis=0)

if len(stack.shape) == 4:
    stack = stack[:, :, :, 0]

plot_validation(stack, segmentation, average_scores, errors,
                model_name + ' - averaged validation scores and pixel errors')
