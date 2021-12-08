import numpy as np


def dtw(S, T):
    def euclidian(t,s):
        return np.sqrt(np.square(t-s))
    sD = S.shape
    tD = T.shape
    N = sD[0]
    M = tD[0]
    if len(sD) == 2:
        sN = sD[1]
    else:
        sN = 1
        S = np.expand_dims(S, axis=1)
    if len(tD) == 2:
        tM = tD[1]
    else:
        tM = 1
        T = np.expand_dims(T, axis=1)
    assert (sN == tM), "S and T 2nd dimension (features) do not match."
    distance = np.zeros((M,N))
    for n in range(N):
        for m in range(M):
            d = euclidian(S[n,:],T[m,:])
            distance[m,n] = np.sum(d)
    cost = np.zeros((M,N))
    cost[0,:] = distance[0,:]
    cost[:,0] = np.cumsum(distance[:,0])
    for n in range(1,N):
        for m in range(1,M):
            cost[m,n] = distance[m,n] + min(cost[m-1, n  ],    # left
                                            cost[m  , n-1],    # below
                                            cost[m-1, n-1])    # left-below
    return distance, cost

def warping_path(cost):
    dist_func_delta = cost[-1,:]
    #plt.plot(dist_func_delta)
    #plt.show()

    i = np.array([-1, 0, -1])
    j = np.array([0, -1, -1])

    c_l = np.argmin(dist_func_delta)
    print(c_l)
    r_l = cost.shape[0]-1
    print(r_l)
    input()
    path_r = [r_l]
    path_c = [c_l]
    while r_l > 0:
        ci = r_l + i
        cj = c_l + j
        ind = np.argmin([cost[ci[0], cj[0]], cost[ci[1], cj[1]], cost[ci[2], cj[2]]])
        r_l += i[ind]
        c_l += j[ind]
        path_r.append(r_l)
        path_c.append(c_l)
        print(len(path_r), len(path_c))
    return [path_r, path_c]

def disparity(imageA, imageB):
    dmap = np.zeros(imageA.shape[:2])

    img_h, img_w = imageA.shape[:2]

    for y in range(img_h): # for each row if image A
        print(y, "of", img_h-1)
        T = imageA[y,:].copy()
        S = imageB[y,:].copy()
        distance, cost = dtw(S, T)
    
        #plt.imshow(distance,origin='lower')
        #plt.colorbar()
        #plt.show()

        #plt.imshow(cost,origin='lower')
        #plt.colorbar()
        #plt.show()

        path = warping_path(cost)
        #plt.imshow(cost,origin='lower')
        #plt.plot(path[1],path[0],'-w')
        #plt.colorbar()
        #plt.show()
        #print(path)
        print(path[0])
        print(path[1])
        dmap[y,:] = np.subtract(path[0],path[1])

    return dmap