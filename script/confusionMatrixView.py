import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, save_name=None, fig_format='tif', title='Confusion Matrix', n_classes=7,
                          x_label='Predict label', y_label='Actual label', precision=2, color='red', axes_ticks=None,
                          text_fontsize=10, x_rotation=90):
    classes = [str(i) for i in range(n_classes)]
    plt.figure(figsize=(12, 8), dpi=300)
    np.set_printoptions(precision=precision)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color=color, fontsize=text_fontsize, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap='cool')
    if title is not None:
        plt.title(title)
    plt.colorbar()
    if axes_ticks is None:
        axes_ticks = classes
    x_locations = np.array(range(len(classes)))
    plt.xticks(x_locations, axes_ticks, rotation=x_rotation)
    plt.yticks(x_locations, axes_ticks)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    if save_name is not None:
        plt.savefig(save_name, format=fig_format, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    xx = [[0.12, 0.88], [0.3, 0.7]]
    plot_confusion_matrix(xx, save_name=None, n_classes=2)
