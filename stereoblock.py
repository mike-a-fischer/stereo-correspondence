import numpy as np

def ssd(A,B):
    return ((A - B)**2).sum()

def disparity(imageA, imageB, window_size=11, disparity_range=50):
    half_window_size = window_size//2
    dmap = np.zeros(imageA.shape[:2])

    img_h, img_w = imageA.shape[:2]

    for y in range(img_h): # for each row if image A
        y1 = max(0, y - half_window_size)
        y2 = min(img_h-1, y + half_window_size)
        for xA in range(img_w):
            x1 = max(0, xA - half_window_size)
            x2 = min(img_w-1, xA + half_window_size)
            windowA = imageA[y1:y2, x1:x2].copy()
            scanline_ssd = []
            scanline_xB  = []
            for d in range(0, int(disparity_range)+1):
                xB1 = x1 + d #max(0, xB - half_window_size)
                xB2 = xB1 + windowA.shape[1] #min(img_w-1, xB + half_window_size)
                if xB1 >= 0 and xB2 < imageB.shape[1]:
                    windowB = imageB[y1:y2, xB1:xB2].copy()
                    scanline_ssd.append(ssd(windowA, windowB))
                    scanline_xB.append(d)
            min_ind = np.argmin(scanline_ssd)
            dmap[y,xA] = scanline_xB[min_ind]# - x1
    return dmap
