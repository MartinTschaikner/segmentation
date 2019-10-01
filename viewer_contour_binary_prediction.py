import matplotlib.pyplot as plt
import numpy as np


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
    seg = ax[0].seg
    seg1 = ax[1].seg
    volume_data = ax[0].volume
    volume_label = ax[1].volume
    ax[0].index = (ax[0].index + 1) % volume_data.shape[0]
    for coll in seg.collections:
        coll.remove()
    for coll in seg1.collections:
        coll.remove()
    img1.set_array(volume_data[ax[0].index, :, :])
    img2.set_array(volume_label[ax[0].index, :, :])
    ax[1].seg = ax[1].contour(volume_label[ax[0].index, :, :], levels=[0.99, 1.99, 2.99], colors='blue', linestyles='-')
    ax[0].seg = ax[0].contour(volume_data[ax[0].index, :, :], levels=0, colors='blue', linestyles='-')
    ax[0].text(12, 40, str(ax[0].index), bbox={'facecolor': 'orange', 'pad': 10})


# Next slice func.
def next_slice_up(ax):
    volume_data = ax[0].volume
    volume_label = ax[1].volume
    seg = ax[0].seg
    seg1 = ax[1].seg
    if ax[0].index >= 1:
        for coll in seg.collections:
            coll.remove()
        for coll in seg1.collections:
            coll.remove()
        ax[0].index = (ax[0].index - 1) % volume_data.shape[0]
        img1.set_array(volume_data[ax[0].index, :, :])
        img2.set_array(volume_label[ax[0].index, :, :])
        ax[1].seg = ax[1].contour(volume_label[ax[0].index, :, :], levels=[0.99, 1.99, 2.99], colors='blue',
                                  linestyles='-')
        ax[0].seg = ax[0].contour(volume_data[ax[0].index, :, :], levels=0, colors='blue', linestyles='-')
        ax[0].text(12, 40, str(ax[0].index), bbox={'facecolor': 'orange', 'pad': 10})


def plot_data_label(volume_data, volume_label, device):
    global img1, img2
    fig, ax = plt.subplots(ncols=2)
    ax[0].volume = volume_data
    num_cols = np.size(volume_data, 2)
    num_rows = np.size(volume_data, 1)
    ax[0].index = 0
    ax[0].seg = ax[0].contour(volume_data[ax[0].index, :, :], levels=0, colors='blue', linestyles='-')
    fig.canvas.mpl_connect('scroll_event', mouse_scroll)
    img1 = ax[0].imshow(volume_data[ax[0].index, :, :], cmap='gray', vmin=0, vmax=1,
                        extent=(-0.5, num_cols - 0.5, num_rows, 0))
    ax[0].set_title(device + ' Label with contour', pad=22)
    ax[0].title.set_size(25)
    ax[0].set_xlabel('number of A scans [ ]', labelpad=18)
    ax[0].xaxis.label.set_size(20)
    ax[0].set_ylabel('Z axis [ ]', labelpad=18)
    ax[0].yaxis.label.set_size(20)

    ax[1].volume = volume_label
    ax[1].seg = ax[1].contour(volume_label[ax[0].index, :, :], levels=[0.99, 1.99, 2.99], colors='blue', linestyles='-')
    img2 = ax[1].imshow(volume_label[ax[0].index, :, :], cmap='gray', vmin=0, vmax=3,
                        extent=(-0.5, num_cols - 0.5, num_rows, 0))
    ax[1].set_title(device + ' ring scan label', pad=22)
    ax[1].title.set_size(25)
    ax[1].set_xlabel('number of A scans [ ]', labelpad=18)
    ax[1].xaxis.label.set_size(20)
    ax[1].yaxis.set_major_locator(plt.NullLocator())

    plt.show()
