import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


def score_text(average_scores, errors):

    average_scores = average_scores.astype('str')
    errors = errors.astype('str')
    # placeholder
    score_string = np.zeros(np.size(average_scores, 1)).astype('U256')
    errors_string = np.zeros(np.size(average_scores, 1)).astype('U256')

    for i in range(np.size(average_scores, 1)):
        string = 'dice:\t\t    ' + average_scores[0, i][:6] + '\n' + \
                 'accuracy:\t' + average_scores[1, i][:6] + '\n' + \
                 'precision:\t ' + average_scores[2, i][:6] + '\n' + \
                 'specicifity:\t ' + average_scores[3, i][:6] + '\n' + \
                 'sensitivity:\t ' + average_scores[4, i][:6] + '\n' + \
                 'balanced acc: ' + average_scores[5, i][:6] + '\n' + \
                 'jaccard: ' + average_scores[6, i][:6] + '\n' + \
                 'tversky7: ' + average_scores[7, i][:6]

        score_string[i] = string.expandtabs()

        string = 'signed/unsigned ILM:\t  ' + errors[0, 0, i][:6] + '/' \
                 + errors[0, 1, i][:6] + '\n' + \
                 'signed/unsigned RNFL:\t' + errors[1, 0, i][:6] + '/'\
                 + errors[1, 1, i][:6] + '\n' + \
                 'signed/unsigned RPE:\t ' + errors[2, 0, i][:6] + '/'\
                 + errors[2, 1, i][:6]

        errors_string[i] = string.expandtabs()

    return score_string, errors_string


# Mouse scroll event.
def mouse_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes
    if event.button == 'down':
        next_slice(ax)
    elif event.button == 'up':
        next_slice_up(ax)
    fig.canvas.draw()


# Next slice func.
def next_slice(ax):
    ax = ax[0]
    stack = ax.stack
    segmentation = ax.segmentation
    scores_string = ax.scores_string
    errors_string = ax.errors_string
    ax.index = (ax.index + 1) % stack.shape[0]
    img.set_array(stack[ax.index, :, :])
    for i in range(3):
        seg_cnn[i].set_ydata(segmentation[ax.index, i, :, 0])
        seg_gt[i].set_ydata(segmentation[ax.index, i, :, 1])

    text_e.set_text(errors_string[ax.index])
    text_s.set_text(scores_string[ax.index])
    ax.text(12, 40, str(ax.index), bbox={'facecolor': 'orange', 'pad': 10})


# Next slice func.
def next_slice_up(ax):
    ax = ax[0]
    stack = ax.stack
    segmentation = ax.segmentation
    scores_string = ax.scores_string
    errors_string = ax.errors_string
    if ax.index >= 1:
        ax.index = (ax.index - 1) % stack.shape[0]
        img.set_array(stack[ax.index, :, :])
        for i in range(3):
            seg_cnn[i].set_ydata(segmentation[ax.index, i, :, 0])
            seg_gt[i].set_ydata(segmentation[ax.index, i, :, 1])

        text_e.set_text(errors_string[ax.index])
        text_s.set_text(scores_string[ax.index])
        ax.text(12, 40, str(ax.index), bbox={'facecolor': 'orange', 'pad': 10})


def plot_validation(stack, segmentation, average_scores, errors, title):
    global img, seg_cnn, seg_gt, text_e, text_s

    fig, ax = plt.subplots()
    ax.stack = stack
    ax.segmentation = segmentation
    ax.index = 0
    scores_string, errors_string = score_text(average_scores, errors)
    ax.scores_string = scores_string
    ax.errors_string = errors_string
    num_cols = np.size(stack, 2)
    num_rows = np.size(stack, 1)

    fig.canvas.mpl_connect('scroll_event', mouse_scroll)
    img = ax.imshow(stack[ax.index, :, :], cmap='gray', vmin=0, vmax=255,
                    extent=(-0.5, num_cols - 0.5, num_rows, 0))

    seg_gt = ax.plot(np.transpose(segmentation[ax.index, :, :, 1], (1, 0)), color='yellow', label='gold standard')
    seg_cnn = ax.plot(np.transpose(segmentation[ax.index, :, :, 0], (1, 0)), color='blue', label='segmentation')

    text_e = ax.text(num_cols//4, num_rows - 25, errors_string[ax.index],
                     fontsize=20, bbox={'facecolor': 'white', 'pad': 10})
    text_s = ax.text(num_cols + 20, 100, scores_string[ax.index],
                     fontsize=20, bbox={'facecolor': 'white', 'pad': 10})
    ax.text(12, 40, str(ax.index), bbox={'facecolor': 'orange', 'pad': 10})

    ax.set_title(title, pad=22)
    ax.title.set_size(25)
    ax.set_xlabel('number of A scans [ ]', labelpad=18)
    ax.xaxis.label.set_size(20)
    ax.set_ylabel('Z axis [ ]', labelpad=18)
    ax.yaxis.label.set_size(20)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=25)

    plt.show()
