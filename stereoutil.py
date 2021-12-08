import numpy as np

def get_accuracy(dmap_true, dmap_pred, scale):
    is_correct = np.abs(dmap_true - dmap_pred) <= 2. * scale
    accuracy = is_correct.sum() / float(dmap_true.size)
    return accuracy

def disparity_jet(dmap):
    dmap_color = cv2.applyColorMap(dmap.astype('uint8'), cv2.COLORMAP_JET)
    return dmap_color