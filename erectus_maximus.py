import numpy
import matplotlib.pyplot as plt

with open('input.txt') as fi:
    lines = fi.readlines()

x = []
y= []

for line in lines:
    temp = line.split(" ")
    temp2 = temp[1].split("\n")
    del temp2[1]
    y.append(float(temp2[0]))
    x.append(float(temp[0]))

N = len(x) -1


h=[]



for i in range(len(x)-1):
    h.append(x[i+1]-x[i])

b= numpy.zeros(N*4)
A = numpy.zeros((N*4,4*N))

S= []

def reverse_recursion(index):
    
    a_array = numpy.array([[1,0,0,0],[1,h[index],h[index]**2,h[index]**3 ],[0,1,2*h[index], 3*h[index]**2],[0,0,2,6*h[index]]])

    a_array = numpy.linalg.inv(a_array)

    if(index == N-1):

        b_array = numpy.array([y[index], y[index+1] , 0,0 ])
    else:

        b_array = numpy.array([y[index], y[index+1] , S[0][1],2*S[0][2]])

    result = numpy.matmul(a_array, b_array)
    S.insert(0,numpy.ndarray.tolist(result))


    if (index != 0):
        reverse_recursion(index-1)

reverse_recursion(N-1)

print(S)

#plotting

plt.plot(x,S)
plt.show()


    

    






    

    








