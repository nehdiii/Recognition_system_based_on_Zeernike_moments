import torch
import numpy as np
from math import factorial
import cv2
from mahotas.features import _zernike
from mahotas.center_of_mass import center_of_mass
import math
import cmath
deg = 8
def zernike_moments(im, radius, degree=deg, cm=None):

    zvalues = {}
    if cm is None:
        c0,c1 = center_of_mass(im)
    else:
        c0,c1 = cm

    Y,X = np.mgrid[:im.shape[0],:im.shape[1]]
    P = im.ravel()

    def rescale(C, centre):
        Cn = C.astype(np.double)
        Cn -= centre
        Cn /= radius
        return Cn.ravel()
    Yn = rescale(Y, c0)
    Xn = rescale(X, c1)

    Dn = Xn**2
    Dn += Yn**2
    np.sqrt(Dn, Dn)
    np.maximum(Dn, 1e-9, out=Dn)
    k = (Dn <= 1.)
    k &= (P > 0)

    frac_center = np.array(P[k], np.double)
    frac_center = frac_center.ravel()
    frac_center /= frac_center.sum()
    Yn = Yn[k]
    Xn = Xn[k]
    Dn = Dn[k]
    An = np.empty(Yn.shape, np.complex_)
    An.real = (Xn/Dn)
    An.imag = (Yn/Dn)

    Ans = [An**p for p in range(2,degree+2)]
    Ans.insert(0, An) # An**1
    Ans.insert(0, np.ones_like(An)) # An**0
    for n in range(degree+1):
        for l in range(n+1):
            if (n-l)%2 == 0:
                z = _zernike.znl(Dn, Ans[l], frac_center, n, l)
                zvalues[(n,l)] = z
    return zvalues


def Improved_Zernike_Moments(outline,moments):
    # calculate the zeroth order geometric moment
    zeroth_order_raw_moments_m00 = cv2.moments(outline)['m00']
    # calculate the improved zernike moment for scale invariance
    for i in moments.keys():
        moments[i] = complex(moments[i].real / zeroth_order_raw_moments_m00,
                             moments[i].imag / zeroth_order_raw_moments_m00)
    return moments

#####################################################
################## compute C #######################

def compute_C(ZerVectImD,ZerVectImT,degree=deg):
    pi = math.pi
    part1ofC = 0
    part2ofC = 0
    for p in range(0,degree+1):
        if p%2 == 0:
            print()
            Zp0D = ZerVectImD[(p,0)]
            Zp0T = ZerVectImT[(p,0)]
            Zp0DT = Zp0D*Zp0D
            Zp0DNormSquare = abs(Zp0D)**2
            Zp0TNormSquare = abs(Zp0T)**2
            Zp0DPhi = cmath.phase(Zp0D)
            Zp0TPhi = cmath.phase(Zp0T)
            Zp0DTNorm= abs(Zp0DT)

            part1ofC += (1/1+p)*(Zp0DNormSquare + Zp0TNormSquare - 2*Zp0DTNorm*math.cos(Zp0TPhi-Zp0DPhi))
    part1ofC *= pi

    for q in range(1, degree + 1):
        for p in range(q, degree + 1):
            if (p - q) % 2 == 0:
                ZpqD = ZerVectImD[(p,q)]
                ZpqT = ZerVectImT[(p,q)]
                ZpqDNormSquare = abs(ZpqD)**2
                ZpqTNormSquare = abs(ZpqT)**2
                part2ofC = (1/1+p)*(ZpqDNormSquare+ZpqTNormSquare)
    part2ofC *= 2*pi
    return part1ofC + part2ofC
######################## compute Aq ####################
def compute_Aq(ZerVectImD,ZerVectImT,q,degree=deg):
    Aq = 0
    for p in range(q,degree+1):
        if (p,q) not in ZerVectImD.keys():
            continue
        else:
            Aq += (1/1+p)*(ZerVectImT[(p,q)].real*ZerVectImD[(p,q)].real + ZerVectImT[(p,q)].imag*ZerVectImD[(p,q)].imag)
    return Aq

def compute_Bq(ZerVectImD,ZerVectImT,q,degree=deg):
    Bq= 0
    for p in range(q,degree+1):
        if (p, q) not in ZerVectImD.keys():
            continue
        else:
            Bq += (1/1+p)*(ZerVectImD[(p,q)].real*ZerVectImT[(p,q)].imag - ZerVectImT[(p,q)].real*ZerVectImD[(p,q)].imag)
    return Bq

def comute_Aq_Bq_overq(ZerVectImD,ZerVectImT,degree=deg):
    Aq_vect,Bq_vect = [],[]
    for q in range(1,degree+1):
        Aq_vect.append(compute_Aq(ZerVectImD,ZerVectImT,q))
        Bq_vect.append(compute_Bq(ZerVectImD,ZerVectImT,q))
    return Aq_vect,Bq_vect



def regual_falsi(f,x1,x2,ZerVectImD,ZerVectImT,tol=1.0e-6,maxfpos=100):
    xh = 0
    fpos = 0
    if f(ZerVectImD,ZerVectImT,x1)*f(ZerVectImD,ZerVectImT,x2)<0:
        for fpos in range(1,maxfpos+1):
            xh = x2 - (x2-x1)*f(ZerVectImD,ZerVectImT,x2)/(f(ZerVectImD,ZerVectImT,x2)-f(ZerVectImD,ZerVectImT,x1))
            if abs(f(ZerVectImD,ZerVectImT,xh)) < tol: break
            elif f(ZerVectImD,ZerVectImT,x1)*f(ZerVectImD,ZerVectImT,xh) < 0:
                x2 = xh
            else:
                x1 = xh
    else:
        print("no root exists")
    return xh,fpos


def Ovrall_func_d(ZerVectImD,ZerVectImT,phi,degree=deg):
    pi = math.pi
    C = compute_C(ZerVectImD,ZerVectImT)
    Aq,Bq = comute_Aq_Bq_overq(ZerVectImD,ZerVectImT)
    sum = 0
    for q in range(len(Aq)):
        sum += Aq[q]*math.cos((q+1)*phi) - Bq[q]*math.sin((q+1)*phi)
    return C - 4*pi*sum

def Overall_derv_func_d(ZerVectImD,ZerVectImT,phi,degree=deg):
    pi = math.pi
    Aq, Bq = comute_Aq_Bq_overq(ZerVectImD, ZerVectImT)
    sum = 0
    for q in range(len(Aq)):
        sum+= q*(Aq[q]*math.sin((q+1)*phi) + Bq[q]*math.cos((q+1)*phi))
    return 4*pi*sum


