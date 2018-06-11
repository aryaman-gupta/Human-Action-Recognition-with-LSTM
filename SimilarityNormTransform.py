
# coding: utf-8

# In[1]:


import numpy as np
import math
import scipy as sp


def SimilarityNormTransform(Pin, Single=False):
    #  similarity normalization transform
    #  Pin = [p1 p2 p3 p4]; 4x3 matrix with four 3D points
    #  Pout is the similarity normalization transform of Pin with respect
    #  to p1,p2, so that p1 goes to (0,0,0) and p2 goes to (1,1,1).
    #
    #  The order of p1,p2,p3,p4 assumes that p1,p2 is the most widely separated
    #  pair of points and p1 is the closest to the camera. Also, between p3 and
    #  p4, p3 is the nearest to p1.

    P = Pin
    T = -1 * P[[0, 0, 0], :]
    p2 = np.subtract(P[1, :], P[0, :])
    theta1 = math.atan2(p2[1], p2[0])
    phiXY1 = -theta1 + sp.pi / 4
    c1 = math.cos(phiXY1)
    s1 = math.sin(phiXY1)
    Rz1 = np.array(([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]]))
    p2_1 = c1 * p2[0] - s1 * p2[1]
    p2[1] = s1 * p2[0] + c1 * p2[1]
    p2[0] = p2_1
    no2 = np.sum(p2[:2] * p2[:2])
    no = 2 * (no2 + p2[2] * p2[2])
    pp = sum(p2[:2]) / math.sqrt(no)
    if pp > 1:
        #        print 'we have cos angle greater than 1.0 and it is ', pp
        pp = 1.0
    phi = math.acos(pp)
    if p2[2] > 0:
        phi = -phi + 0.615479708670387
    else:
        phi = phi + 0.615479708670387

    r = np.array([p2[1], -p2[0], 0])
    r = r / math.sqrt(no2)

    C = math.cos(phi)
    S = math.sin(phi)
    F = 1 - C

    RR = np.array(([[F * pow(r[0], 2) + C, F * r[0] * r[1], S * r[1]],
                    [F * r[0] * r[1], F * pow(r[1], 2) + C, -S * r[0]],
                    [-S * r[1], S * r[0], C]]))
    P = P[1:, :]
    P = np.transpose(P + T)
    P = RR.dot(Rz1.dot(P))
    P = P / P[0, 0]
    if Single:
        QuadDescriptor = np.array([P[0, 1], P[1, 1], P[2, 1], P[0, 2], P[1, 2], P[2, 2]])
    else:
        N1 = math.sqrt(np.sum(P[:, 1] * P[:, 1]))
        N2 = math.sqrt(np.sum(P[:, 2] * P[:, 2]))
        #     print 'norm1',N1,'norm2',N2
        if N1 <= N2:
            QuadDescriptor = np.array([P[0, 1], P[1, 1], P[2, 1], P[0, 2], P[1, 2], P[2, 2]])
        else:
            QuadDescriptor = np.array([P[0, 2], P[1, 2], P[2, 2], P[0, 1], P[1, 1], P[2, 1]])
    return QuadDescriptor

