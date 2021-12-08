import numpy as np
import cv2
import maxflow
import heapq
from tqdm import tqdm
import multiprocessing as mp

NUM_CPUS = max(1, mp.cpu_count()-1)

def disparity(left, right):
    global run_disp
    left_split = np.array_split(left, NUM_CPUS)
    right_split = np.array_split(right, NUM_CPUS)
    with mp.Pool(processes=NUM_CPUS) as pool:
        disparity = pool.starmap(run_disp, zip(left_split,right_split))
    disparity = np.vstack(disparity)
    is_occluded = disparity >= 1<<30
    disparity[is_occluded] = 0
    return disparity

def run_disp(left, right):
    stereograph = StereoGraphCut(left, right)
    disparity = stereograph.run()
    return disparity

class StereoGraphCut:
    VAR_ALPHA = -1
    VAR_ABSENT = -2
    CUTOFF = 30
    OCCLUDED = 1<<30

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.img_h, self.img_w = self.left.shape[:2]
        self.img_size = self.left.size
        
        self.max_iterations = 4
        self.alpha_range = 16
        
        self.energy = 0
        
        self.vars0 = np.zeros((self.img_h, self.img_w), dtype=np.int64)
        self.varsA = np.zeros((self.img_h, self.img_w), dtype=np.int64)
        
        self.active_penalty = 0
        self.edge_thresh = 8
        
        self.disparity = np.full((self.img_h, self.img_w), self.OCCLUDED, dtype=np.int64)
        
    def run(self):
        print("Running...")
        self.K, self.lambda1, self.lambda2, self.denominator = self.generate_K()
        num_disparity = self.alpha_range + 1
        self.energy = 0
        done = np.full(num_disparity, False)
        for _ in tqdm(range(self.max_iterations), total=self.max_iterations):
            for label in np.random.permutation(num_disparity):
                if not done[label]:
                    self.active_penalty = 0
                    g = maxflow.Graph[int](2*self.img_size, 12*self.img_size)
                    #print('Adding data occlusion terms for', label)
                    self.add_data_occlusion_terms(g, label)
                    #print('Adding smoothness terms for', label)
                    self.add_smoothness_terms(g, label)
                    #print('Adding uniqueness terms for', label)
                    self.add_uniqueness_terms(g, label)
                    
                    new_energy = g.maxflow() + self.active_penalty
                    if new_energy < self.energy:
                        self.update_disparity(g, label)
                        self.energy = new_energy
                        done[:] = False
                    done[label] = True
                    if np.all(done):
                        return self.disparity
        return self.disparity
    
    def generate_K(self):
        k = max(3, (self.alpha_range+3) // 4)
        total_smallest = 0
        total = self.img_h * (self.img_w - self.alpha_range)
        print((self.img_h, self.img_w - self.alpha_range))
        for row, col in tqdm(np.ndindex((self.img_h, self.img_w - self.alpha_range)), total=(self.img_h * (self.img_w - self.alpha_range))):
            deltas = []
            for d in range(0, self.alpha_range+1):
                delta = -1 * self.get_distance((row, col), (row, col + d))
                heapq.heappush(deltas, delta)
                if len(deltas) -1 >= k:
                    heapq.heappop(deltas)
            total_smallest += -1 * heapq.heappop(deltas)
        K = total_smallest/total

        lambda2 = K / 5
        lambda1 = 3 * lambda2

        N = []
        for i in range(1,17):
            first_numerator = round(i * K)
            first = abs(first_numerator / (i * K) - 1)
            second_numerator = round(i * lambda1)
            second = abs(second_numerator / (i * lambda1) - 1)
            third_numerator = round(i * lambda2)
            third = abs(third_numerator / (i * lambda2) - 1)
            N.append((first+second+third, first_numerator, second_numerator, third_numerator, i))
            
        N = sorted(N)
        _, K, lambda1, lambda2, denominator = N[0]
        return K, lambda1, lambda2, denominator
    
    def add_data_occlusion_terms(self, g, label):
        for L, _ in np.ndenumerate(self.left):

            disp = self.disparity[L]
            R = (L[0], L[1] + disp)

            if disp == label:
                self.vars0[L] = self.VAR_ALPHA
                self.varsA[L] = self.VAR_ALPHA
                penalty = self.denominator * self.get_distance(L, R) - self.K
                self.active_penalty += penalty
            else:
                #vars0
                if disp != self.OCCLUDED:
                    penalty = self.denominator * self.get_distance(L, R) - self.K
                    node_id = g.add_nodes(1)[0]
                    g.add_tedge(node_id, 0, penalty)
                    self.vars0[L] = node_id
                else:
                    self.vars0[L] = self.VAR_ABSENT

                #varsA
                R = (L[0], L[1] + label)
                if isValidLoc(R, (self.img_h, self.img_w)):
                    penalty = self.denominator * self.get_distance(L, R) - self.K
                    node_id = g.add_nodes(1)[0]
                    g.add_tedge(node_id, penalty, 0)
                    self.varsA[L] = node_id
                else:
                    self.varsA[L] = self.VAR_ABSENT
    
    def add_smoothness_terms(self, g, label):
        for L, _ in np.ndenumerate(self.left):
            for i in range(2):
                L2 = (L[0]+i, L[1]+i-1)
                if isValidLoc(L2, (self.img_h, self.img_w)):

                    disp1 = self.disparity[L]
                    vars01 = self.vars0[L]

                    disp2 = self.disparity[L2]
                    vars02 = self.vars0[L2]

                    # pairwise assignment
                    if disp1 == disp2 and vars01 >= 0 and vars02 >= 0:
                        delta = self.get_disp_distance(L, L2, disp1)
                        addPairwise(g, vars01, vars02, 0, delta, delta, 0)

                    # vars01
                    if disp1 != disp2 and vars01 >= 0 and isValidLoc((L2[0], L2[1] + disp1), (self.img_h, self.img_w)):
                        g.add_tedge(vars01, 0, self.get_disp_distance(L, L2, disp1))

                    # vars02
                    if disp1 != disp2 and vars02 >= 0 and isValidLoc((L[0], L[1] + disp2), (self.img_h, self.img_w)):
                        g.add_tedge(vars02, 0, self.get_disp_distance(L, L2, disp2))
                        
                    varsA1 = self.varsA[L]
                    varsA2 = self.varsA[L2]
                    
                    # varsA1 and varsA2
                    if varsA1 != self.VAR_ABSENT and varsA2 != self.VAR_ABSENT:
                        delta = self.get_disp_distance(L, L2, label)
                        if varsA1 != self.VAR_ALPHA:
                            if varsA2 != self.VAR_ALPHA:
                                addPairwise(g, varsA1, varsA2, 0, delta, delta, 0)
                            else:
                                g.add_tedge(varsA1, 0, delta)
                        elif varsA2 != self.VAR_ALPHA:
                            g.add_tedge(varsA2, 0, delta)
    
    def add_uniqueness_terms(self, g, label):
        for L, _ in np.ndenumerate(self.left):
            if self.vars0[L] >= 0:
                varA = self.varsA[L]
                if varA != self.VAR_ABSENT:
                    forbid01(g, self.vars0[L], varA, self.OCCLUDED)

                disp = self.disparity[L]
                L2 = (L[0], L[1] + disp - label)
                if isValidLoc(L2, (self.img_h, self.img_w)):
                    varA = self.varsA[L2]
                    forbid01(g, self.vars0[L], varA, self.OCCLUDED)
                    
    def get_distance(self, p, q):
        p_dist = 0
        q_dist = 0
        left_P = self.left[p]
        left_min, left_max = subPixel(self.left)
        left_P_min = left_min[p]
        left_P_max = left_max[p]
        
        right_Q = self.right[q]
        right_min, right_max = subPixel(self.right)
        right_Q_min = right_min[q]
        right_Q_max = right_max[q]

        
        if left_P < right_Q_min:
            p_dist = right_Q_min - left_P
        elif left_P > right_Q_max:
            p_dist = left_P - right_Q_max
            
        if right_Q < left_P_min:
            q_dist = left_P_min - right_Q
        elif right_Q > left_P_max:
            q_dist = right_Q - left_P_max

        d = min(p_dist, q_dist, self.CUTOFF)
        return d**2
    
    def get_disp_distance(self, L, L2, disp):
        left_dist = abs(self.left[L] - self.left[L2])
        right_dist = abs(self.right[L[0], L[1] + disp] - self.right[L2[0], L2[1] + disp])

        if left_dist < self.edge_thresh and right_dist < self.edge_thresh:
            return self.lambda1
        return self.lambda2
    
    def update_disparity(self, g, label):
        vecGetSegment = np.vectorize(g.get_segment)

        if self.vars0[self.vars0 >= 0].size > 0:
            vars0Segments = np.zeros(self.vars0.shape)
            vars0Segments[self.vars0 >= 0] = vecGetSegment(self.vars0[self.vars0 >= 0])
            self.disparity[vars0Segments == 1] = self.OCCLUDED

        if self.varsA[self.varsA >= 0].size > 0:
            varsASegments = np.zeros(self.varsA.shape)
            varsASegments[self.varsA >= 0] = vecGetSegment(self.varsA[self.varsA >= 0])
            self.disparity[varsASegments == 1] = label
    
def subPixel(im):
    left = np.copy(im).astype(float)
    left[:,1:] += im[:,:-1]
    left[:,1:] /= 2

    right = np.copy(im).astype(float)
    right[:,:-1] += im[:,1:]
    right[:,:-1] /=2

    up = np.copy(im).astype(float)
    up[1:,:] += im[:-1,:]
    up[1:,:] /=2

    down = np.copy(im).astype(float)
    down[:-1,:] += im[1:,:]
    down[:-1,:] /=2
    
    minimum = np.array([im, left, right, up, down]).min(axis=0)
    maximum = np.array([im, left, right, up, down]).max(axis=0)
    return minimum, maximum

def isValidLoc(coordP, coordR):
    rowValid = 0 <= coordP[0] < coordR[0]
    colValid = 0 <= coordP[1] < coordR[1]
    return rowValid and colValid

def addPairwise(g, n1, n2, E00, E01, E10, E11):
    g.add_edge(n1, n2, 0, (E01+E10)-(E00+E11))
    g.add_tedge(n1, E11, E01)
    g.add_tedge(n2, 0, E00 - E01)

def forbid01(g, n1, n2, OCCLUDED):
    g.add_edge(n1, n2, OCCLUDED, 0)