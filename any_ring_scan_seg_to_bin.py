from keras.models import load_model
from prediction_stitching_to_seg_to_bin import PredictionStitchingToSegmentation
from split_resize_scan_to_windows import *
from losses import *
import return_4D_ring_scan_stack
from tkinter.filedialog import askopenfilename

# interpolation parameters
radius = 1.75
number_circle_points = 768
filter_parameter = 1

# window width, height for model input and stride of window over ring scan image
width = 128
height = 512
stride_x = 64

# import model
save_path = 'SavedNetworks'
model = load_model(save_path + '/model_k5_c8_xe_25.h5', custom_objects={'cce_dice_loss': cce_dice_loss})

# import (interpolated) ring scan (stack) as 4d stack
file = askopenfilename(title='Choose data to apply segmentation!')

input_data = return_4D_ring_scan_stack.Return4DRingScanStack(file)
ring_scan_stack = input_data.return_4d_stack(radius, number_circle_points, filter_parameter)
ring_scan_stack = ring_scan_stack.astype('uint8')

# reduce ring scan stack to single ring scan
if np.size(ring_scan_stack, 0) != 1:
    t = 5
    ring_scan_stack = ring_scan_stack[t:t+1, :, :, :]

# resize and split ring scan into overlapping windows
ring_scan_window_stack = windowing(ring_scan_stack, 'ring scan(s)', width, height, stride_x, 1)
ring_scan_window_stack = ring_scan_window_stack/255
num_windows_per_ring_scan = np.ceil(np.size(ring_scan_window_stack, 0)/np.size(ring_scan_stack, 0)).astype('int')
print(num_windows_per_ring_scan, 'windows per ring scan.')

# compute model prediction for each window
prediction_model = model.predict(ring_scan_window_stack, batch_size=1, verbose=1)

# compute stitched prediction and results
prediction_segmentation = PredictionStitchingToSegmentation(prediction_model, stride_x, np.size(ring_scan_stack, 1))
stitched_prediction = prediction_segmentation.stitching_prediction(filter_type='gaussian')
all_segmentation = prediction_segmentation.\
    segmentation_stitched_prediction(stitched_prediction)
stitched_prediction_binary = prediction_segmentation.seg_to_binary_prediction(all_segmentation, stitched_prediction)

# plot results
prediction_segmentation.plot_stitched_prediction(stitched_prediction)
prediction_segmentation.plot_binaries(stitched_prediction_binary)
prediction_segmentation.plot_segmentation(ring_scan_stack, all_segmentation, [])
