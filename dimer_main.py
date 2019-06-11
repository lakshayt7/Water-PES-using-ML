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
for i in range(100):
    f = open("water" + str(i) + ".log", "rt")
    contents = f.read()
    f.close()
    d = contents.find('Dipole')
    i = contents.find('X', d)
    mono = ''
    Dx.append(number(contents, i))
    i = contents.find('Y', d)
    mono = ''
    Dy.append(number(contents, i))
    i = contents.find('Z', d)
    mono = ''
    Dz.append(number(contents, i))
print(Dy)


