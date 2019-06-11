import numpy as np
def number(s, i):
   while((s[i]<'0' or s[i]>'9') and s[i] != '-'):
	i+=1
   mono = ''
   while(s[i] == '-' or s[i] == '.' or ('0' <= s[i] and s[i] <= '9')):
       mono = mono + s[i]
       i+=1
   return mono
n = 5
key = ['X', 'Y', 'Z']
Dx, Dy, Dz = [], [], []
Qxx, Qyy, Qzz, Qxy, Qxz, Qyz = [], [], [], [], [], []
Qxxx, Qyyy, Qzzz, Qxyy, Qxxy, Qxxz, Qxzz, Qyzz, Qyyz, Qxyz = [], [], [], [], [], [], [], [], [], []

for i in range(100):
    f = open("water" + str(i) + ".log", "rt")
    contents = f.read()
    f.close()
    d = contents.find('Dipole')
    i = contents.find('X', d)
    Dx.append(number(contents, i))
    i = contents.find('Y', d)
    Dy.append(number(contents, i))
    i = contents.find('Z', d)
    Dz.append(number(contents, i))
    d = contents.find('Quadrupole')
    i = contents.find('XX', d)
    Qxx.append(number(contents, i))
    i = contents.find('YY', d)
    Qyy.append(number(contents, i))
    i = contents.find('ZZ', d)
    Qzz.append(number(contents, i))
    i = contents.find('XY', d)
    Qxy.append(number(contents, i))
    i = contents.find('XZ', d)
    Qxz.append(number(contents, i))
    i = contents.find('YZ', d)
    Qyz.append(number(contents, i))
    d = contents.find('Octapole')
    i = contents.find('XXX', d)
    Qxxx.append(number(contents, i))
    i = contents.find('YYY', d)
    Qyyy.append(number(contents, i))
    i = contents.find('ZZZ', d)
    Qzzz.append(number(contents, i))
    i = contents.find('XYY', d)
    QXXY.append(number(contents, i))
    i = contents.find('XXY', d)
    Qxxy.append(number(contents, i))
    
print(Qxy)


