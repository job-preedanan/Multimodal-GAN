import numpy as np
import cv2


def plot_summary_graph(history, save_path):
    import matplotlib.pyplot as plt

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_loss.png')
    plt.clf()

    # summarize history for accuracy
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice coefficient')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_acc.png')
    plt.clf()

    # summarize history for recall
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_recall.png')
    plt.clf()

    # summarize history for precision
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_precision.png')
    plt.clf()


def convert_to_heatmap(org_image, predict, hif=1.0):
    org_image = np.stack((org_image,) * 3, axis=-1)
    # heat map generate
    predict = np.uint8(255 * (predict/np.max(predict)))
    predict = cv2.applyColorMap(predict, cv2.COLORMAP_JET)
    heatmap = predict * hif + org_image

    return heatmap


def normalize_x(image):
    return image / 127.5 - 1


def denormalize_x(image):
    return np.array((image + 1) * 127.5, dtype=np.uint8)


def normalize_y(image):
    return image / 255


def denormalize_y(image):
    return image * 255


