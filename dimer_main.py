import numpy as np
n = 5
for i in range(5):
    f = open("water.log", "rt")
    contents = f.read()
    f.close()
    print(contents)
    i = contents.find('Monopole = ')
    mono = ''
    while('0' <= contents[i+11] <='9'):
        mono = mono + contents[i+11]
        i+=1
    
