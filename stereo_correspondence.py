import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from load_middlebury import read_calib, create_depth_map, read_pfm

import stereoblock
import stereodp
import stereograph
import stereoutil

def load_files(stereo_folder, scale=0.1):
    right = cv2.imread(os.path.join(stereo_folder, 'im0.png'),0)
    left = cv2.imread(os.path.join(stereo_folder, 'im1.png'),0)

    left_small = cv2.resize(left, None, fx=scale, fy=scale).astype(np.float32)
    right_small = cv2.resize(right, None, fx=scale, fy=scale).astype(np.float32)

    calib_file_path = os.path.join(stereo_folder, 'calib.txt')
    calib = read_calib(calib_file_path)

    dmap_truth, [(height, width, channels), _] = read_pfm(os.path.join(stereo_folder, 'disp1.pfm'))
    dmap_truth = np.reshape(dmap_truth, newshape=(height, width, channels))
    dmap_truth = np.flipud(dmap_truth)
    dmap_truth_small = cv2.resize(dmap_truth, None, fx=scale, fy=scale).astype('float')
    where_are_finite = np.isfinite(dmap_truth_small)
    dmap_truth_small[~where_are_finite] = 0.0
    dmap_truth_small *= scale

    return left_small, right_small, dmap_truth_small, calib
    

def match_stereo_pairs(stereo_folder, scales=[0.1], output=True):
    block_durations = []
    block_accuracy  = []
    graph_durations = []
    graph_accuracy = []
    for scale in scales:
        left, right, dmap_truth, calib = load_files(stereo_folder, scale=scale)
        min_truth = np.min(dmap_truth)
        max_truth = np.max(dmap_truth)

        start_time = time.time()
        dmap_block = stereoblock.disparity(left, right, window_size=11, disparity_range=round(max_truth,-1))
        block_durations.append(time.time()-start_time)
        block_accuracy.append(stereoutil.get_accuracy(dmap_truth, dmap_block, scale))
        print("Block Match took {} seconds with {} accuracy.".format(block_durations[-1], block_accuracy[-1]))

        start_time = time.time()
        dmap_graph = stereograph.disparity(right, left)
        graph_durations.append(time.time()-start_time)
        graph_accuracy.append(stereoutil.get_accuracy(dmap_truth, dmap_graph, scale))
        print("Graph Cut took {} seconds with {} accuracy.".format(graph_durations[-1], graph_accuracy[-1]))

    return block_accuracy, graph_accuracy, block_durations, graph_durations

if __name__ == "__main__":
    stereo_folers = ['./input/Adirondack-perfect',
                     './input/Motorcycle-perfect',
                     './input/Pipes-perfect',
                     './input/Jadeplant-perfect']
    scales = [0.05, 0.1, 0.2]
    for stereo_foler in stereo_folers:
        block_accuracy, graph_accuracy, block_durations, graph_durations = match_stereo_pairs(stereo_foler, scales=scales)
