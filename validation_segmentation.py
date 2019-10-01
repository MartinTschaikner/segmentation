from keras.models import load_model
from prediction_stitching_to_seg_to_bin import PredictionStitchingToSegmentation
from split_resize_scan_to_windows import *
from losses import *
from keras.utils import np_utils
from validation_scores import *
from segmentation_countor_to_array import contour_to_array
from tkinter.filedialog import askdirectory
from os import walk

# window width, height for model input and stride of window over ring scan image
width = 128
height = 512
stride_x = 64

save_path = 'SavedNetworks/'
results_save_dir = 'validation/'

dir_models = askdirectory(title='Choose directory of saved models to compute corresponding scores!')
(_, _, models) = next(walk(dir_models))

# load ring scans and labels for validation
ring_scan_stack = np.load(r'data\validation_Spectralis_ring_scan\ring_scan_stack.npy')[:, :, :, :]
y_test = np.load(r'data\validation_Spectralis_ring_scan\ring_scan_label_stack.npy')[:, :, :, :]
Y_test = np_utils.to_categorical(y_test, num_classes=4, dtype=np.uint8)


for m in range(len(models)):
    model_name = models[m]
    model_file_name = dir_models + '/' + models[m]
    model_dir = model_name[:-6]

    if model_dir.find('dice') != -1:
        custom = {'cce_dice_loss': cce_dice_loss}
    elif model_dir.find('tversky') != -1:
        custom = {'cce_tvesky_loss7': cce_tvesky_loss7}
    elif model_dir.find('jaccard') != -1:
        custom = {'cce_jaccard_loss': cce_jaccard_loss}
    else:
        custom = None

    # import model
    model = load_model(model_file_name, custom_objects=custom)

    # placeholder prediction labels
    Y_prediction = np.empty(Y_test.shape).astype(np.uint8)

    # placeholder scores (num_labels, num_scores, num_scans)
    scores = np.empty((4, 8, np.size(ring_scan_stack, 0)))
    errors = np.empty((3, 2, np.size(ring_scan_stack, 0)))

    # placeholder segmentation 2 channels ground truth and CNN segmentation
    final_segmentation = np.empty((np.size(ring_scan_stack, 0), 3, np.size(ring_scan_stack, 2), 2))

    for i in range(np.size(ring_scan_stack, 0)):
        # prepare model input data by resizing and splitting ring scan into overlapping windows
        ring_scan_window_stack = windowing(ring_scan_stack[i:i+1, :, :, :], 'ring scan(s)', width, height, stride_x, 1)
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
        stitched_prediction_binary = prediction_segmentation.\
            seg_to_binary_prediction(all_segmentation, stitched_prediction)

        # compute ground truth segmentation
        ground_truth_segmentation = prediction_segmentation. \
            segmentation_stitched_prediction(y_test[i, :, :, 0])

        Y_prediction[i, :, :, :] = stitched_prediction_binary

        # plot segmentation
        # prediction_segmentation.plot_stitched_prediction(stitched_prediction)
        # prediction_segmentation.plot_binaries(stitched_prediction_binary)
        # prediction_segmentation.plot_segmentation(ring_scan_stack[i:i+1, :, :, :],
                                                  # all_segmentation,
                                                  # ground_truth_segmentation)

        # transform contour to array of correct length
        segmentation = contour_to_array(all_segmentation, np.size(ring_scan_stack, 2))
        ground_truth = contour_to_array(ground_truth_segmentation, np.size(ring_scan_stack, 2))

        final_segmentation[i, :, :, 0] = segmentation
        final_segmentation[i, :, :, 1] = ground_truth

        errors[:, 0, i], errors[:, 1, i] = un_signed_error(segmentation, ground_truth)

        for k in range(np.size(Y_test, 3)):
            scores[k, 0, i] = dice_sc(Y_test[i, :, :, k], Y_prediction[i, :, :, k])
            scores[k, 1, i] = accuracy(Y_test[i, :, :, k], Y_prediction[i, :, :, k])
            scores[k, 2, i] = precision(Y_test[i, :, :, k], Y_prediction[i, :, :, k])
            scores[k, 3, i] = specificity(Y_test[i, :, :, k], Y_prediction[i, :, :, k])
            scores[k, 4, i] = sensitivity(Y_test[i, :, :, k], Y_prediction[i, :, :, k])
            scores[k, 5, i] = balanced_accuracy(Y_test[i, :, :, k], Y_prediction[i, :, :, k])
            scores[k, 6, i] = jaccard(Y_test[i, :, :, k], Y_prediction[i, :, :, k])
            scores[k, 7, i] = tversky7(Y_test[i, :, :, k], Y_prediction[i, :, :, k])

    np.save(results_save_dir + model_dir + '/' + 'scores.npy', scores)
    np.save(results_save_dir + model_dir + '/' + 'errors.npy', errors)
    np.save(results_save_dir + model_dir + '/' + 'segmentation.npy', final_segmentation)
