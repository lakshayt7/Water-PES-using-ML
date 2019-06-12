import math
import numpy as np
import random
import os
import subprocess

a0 =  0.957835
theta = (104.5/360)*2*math.pi
def RotMatrix(a1, a2, a3) :
    Rx  = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(a1),       -math.sin(a1)       ],
                    [0,         math.sin(a1),        math.cos(a1)       ]])
    Ry =  np.array([[math.cos(a2),          0,      math.sin(a2)        ],
                    [0,                     1,      0                   ],
                    [-math.sin(a2),         0,      math.cos(a2)        ]])
    Rz =  np.array([[math.cos(a3),          -math.sin(a3),          0],
                    [math.sin(a3),          math.cos(a3),           0],
                    [0,                     0,                      1]])
    R = np.dot(Rz, np.dot( Ry, Rx ))
    return R
def H2O2(r, th, ph, alpha):
    O = np.array([r*math.sin(th)*math.cos(ph), r*math.sin(th)*math.sin(ph), r*math.cos(th)])
    H1 = np.array([a0, 0, 0])
    H2 = np.array([a0*math.cos(theta), a0*math.sin(theta), 0])
    rot = RotMatrix(alpha[0], alpha[1], alpha[2])
    H1 = np.dot(rot, H1.T)
    H2 = np.dot(rot, H2.T)
    H1 = H1 + O
    H2 = H2 + O
    return O, H1, H2
def pratom(f, A):
    if(A[0]>=0):
        f.write(" %.5f       " % (A[0]))
    else:
        f.write("%.5f       " % (A[0]))
    if(A[1]>=0):
        f.write(" %.5f       " % (A[1]))
    else:
        f.write("%.5f       " % (A[1]))
    if(A[2]>=0):
        f.write(" %.5f\n" % (A[2]))
    else:
        f.write("%.5f\n" % (A[2]))
O1 = [0,                  0,                  0]
H1 = [a0,                 0,                  0]
H2 = [a0*math.cos(theta), a0*math.sin(theta), 0]
#os.chdir("/home//top//Documents")
n = 5
I = np.zeros((n, 6))
for i in range(n):
    r = random.uniform(1, 8)
    t = random.uniform(0, math.pi)
    p = random.uniform(0, math.pi * 2)
    alpha = [random.uniform(0, math.pi * 2), random.uniform(0, math.pi * 2), random.uniform(0, math.pi * 2)]
    O2, H3, H4 = H2O2(r, t, p, alpha)
    f = open("water%d.com" % (i), "w+")
    f.write("%mem=8GB\n%CHK=water.chk\n#n B3LYP/aug-cc-pVTZ SP\n\n water_2\n\n0 1\n")
    f.write("O         ")
    pratom(f, O1)
    f.write("H         ")
    pratom(f, H1)
    f.write("H         ")
    pratom(f, H2)
    f.write("O         ")
    pratom(f, O2)
    f.write("H         ")
    pratom(f, H3)
    f.write("H         ")
    pratom(f, H4)
    f.write("\n")
    cmd = "G09run water.com"
    #subprocess.call(cmd, shell=True)
    f.close()
    print([r, t, p, alpha[0], alpha[1], alpha[2]])
    I[i] = [r, t, p, alpha[0], alpha[1], alpha[2]]
np.savetxt("Input.csv", I, delimiter = ",")
