import numpy as np
def number(s, i):
   while((s[i]<'0' or s[i]>'9') and s[i] != '-'):
	i+=1
   mono = ''
   while(s[i] == '-' or s[i] == '.' or ('0' <= s[i] and s[i] <= '9')):
       mono = mono + s[i]
       i+=1
   return mono

n = 1000

Distr = ['X', 'Y', 'Z']
Quadstr = ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ'] 
Octstr = ['XXX', 'YYY', 'ZZZ', 'XYY', 'XXY', 'XXZ', 'XZZ', 'YZZ', 'YYZ', 'XYZ']
Hexstr = ['XXXX', 'YYYY', 'ZZZZ', 'XXXY', 'XXXZ' ,'YYYX', 'YYYZ', 'ZZZX', 'ZZZY', 'XXYY', 'XXZZ', 'YYZZ', 'XXYZ', 'YYXZ', 'ZZXY']
Oct = np.zeros((n, 10))
Hex = np.zeros((n, 15))
Quad = np.zeros((n, 6))
Di = np.zeros((n, 3))

for i in range(n):
    f = open("water" + str(i) + ".log", "rt")
    contents = f.read()
    f.close()
    d = contents.find('Dipole')
    for j in range(3):
        l = contents.find(Distr[j], d)
        Di[i][j] = number(contents, l)
    d = contents.find('Quadrupole')
    for j in range(6):
        l = contents.find(Quadstr[j], d)
        Quad[i][j] = number(contents, l)
    d = contents.find('Octapole')
    for j in range(10):
	l = contents.find(Octstr[j], d)
	Oct[i][j] = number(contents, l)
    for j in range(15):
	l = contents.find(Hexstr[j], d)
	Hex[i][j] = number(contents, l)
#np.savetxt("Dipole_Output.csv", Di, delimiter = ",")
#np.savetxt("Quadrapole_Output.csv", Quad, delimiter = ",")
#np.savetxt("Hexadecapole_Output.csv", Hex, delimiter = ",")
#np.savetxt("Octapole_Output.csv", Oct, delimiter = ",")
