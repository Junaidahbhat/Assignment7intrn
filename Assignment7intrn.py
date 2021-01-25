# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 02:15:44 2021

@author: JUNAID AHMAD BHAT
"""
 
import numpy as np
import matplotlib.pyplot as plt
import math


def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB


def circ_gen(B,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + B).T
	return x_circ

#cordinates whould be 

B = np.array([0,0])
C = np.array([10,0])
BC_mid = np.array([5 ,0])

#calculation for the points of contact of the tangents
#the point of contact will be the point of intersection of the circles
#therefore we can obtain them by solving the system of equation of two circles
#the eqution of circles are as
# x^2 + y^2 = 36(given) and (x-5)**2 + y**2 = 5**2(contructed)
#on putting first equation in second equation we get
x_1 = 3.6


#similary for C
x_1 = 3.6

y_1 = math.sqrt(36-((x_1)**2))
y_2 = -math.sqrt(36-((x_1)**2))

 #therefore the coordinates are
print(x_1,y_1)
print(x_1,y_2)

A = np.array([x_1,y_1])
D = np.array([x_1,y_2])


x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_CD = line_gen(C,D)

#given radius of the circle is 6
r = 6
O = np.array([0,0])
x_circ = circ_gen(O,r)
x_circ_OC = circ_gen(OC_mid,5)

plt.plot(x_circ[0,:],x_circ[1,:])
plt.plot(x_circ_OC[0,:],x_circ_OC[1,:])


plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CA[0,:],x_CA[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])


plt.plot(C[0],C[1],'o')
plt.plot(O[0],O[1],'o')
plt.plot(OC_mid[0],OC_mid[1],'o')
plt.plot(A[0],A[1],'o')
plt.plot(D[0],D[1],'o')
plt.text(O[0]*(1+0.1),O[1]*(1+0.1),"B")
plt.text(C[0]*(1+0.1),C[1]*(1+0.1),"C")
plt.text(OC_mid[0]*(1+0.1),OC_mid[1]*(1+0.1),"S")


plt.text(A[0]*(1+0.1),A[1]*(1+0.1),"A")
plt.text(D[0]*(1+0.1),D[1]*(1+0.1),"D")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.axis("equal")
plt.show() 
