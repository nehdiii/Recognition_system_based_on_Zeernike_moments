from mahotas.features import _zernike
from mahotas.center_of_mass import center_of_mass
import math
import cmath




class OptimalSimilartyMeasure:

    def __init__(self,degree):
        self.degree = degree

    def compute_C(self,ZerVectImD, ZerVectImT):
        pi = math.pi
        part1ofC = 0
        part2ofC = 0
        for p in range(0, self.degree + 1):
            if p % 2 == 0:
                print()
                Zp0D = ZerVectImD[(p, 0)]
                Zp0T = ZerVectImT[(p, 0)]
                Zp0DT = Zp0D * Zp0D
                Zp0DNormSquare = abs(Zp0D) ** 2
                Zp0TNormSquare = abs(Zp0T) ** 2
                Zp0DPhi = cmath.phase(Zp0D)
                Zp0TPhi = cmath.phase(Zp0T)
                Zp0DTNorm = abs(Zp0DT)

                part1ofC += (1 / 1 + p) * (
                            Zp0DNormSquare + Zp0TNormSquare - 2 * Zp0DTNorm * math.cos(Zp0TPhi - Zp0DPhi))
        part1ofC *= pi

        for q in range(1, self.degree + 1):
            for p in range(q, self.degree + 1):
                if (p - q) % 2 == 0:
                    ZpqD = ZerVectImD[(p, q)]
                    ZpqT = ZerVectImT[(p, q)]
                    ZpqDNormSquare = abs(ZpqD) ** 2
                    ZpqTNormSquare = abs(ZpqT) ** 2
                    part2ofC = (1 / 1 + p) * (ZpqDNormSquare + ZpqTNormSquare)
        part2ofC *= 2 * pi
        return part1ofC + part2ofC

    def compute_Aq(self,ZerVectImD, ZerVectImT, q):
        Aq = 0
        for p in range(q, self.degree + 1):
            if (p, q) not in ZerVectImD.keys():
                continue
            else:
                Aq += (1 / 1 + p) * (
                            ZerVectImT[(p, q)].real * ZerVectImD[(p, q)].real + ZerVectImT[(p, q)].imag * ZerVectImD[
                        (p, q)].imag)
        return Aq

    def comute_Aq_Bq_overq(self,ZerVectImD, ZerVectImT):
        Aq_vect, Bq_vect = [], []
        for q in range(1, self.degree + 1):
            Aq_vect.append(self.compute_Aq(ZerVectImD, ZerVectImT, q))
            Bq_vect.append(self.compute_Bq(ZerVectImD, ZerVectImT, q))
        return Aq_vect, Bq_vect


    def compute_Bq(self,ZerVectImD, ZerVectImT, q):
        Bq = 0
        for p in range(q, self.degree + 1):
            if (p, q) not in ZerVectImD.keys():
                continue
            else:
                Bq += (1 / 1 + p) * (
                            ZerVectImD[(p, q)].real * ZerVectImT[(p, q)].imag - ZerVectImT[(p, q)].real * ZerVectImD[
                        (p, q)].imag)
        return Bq

    def Ovrall_func_d(self,ZerVectImD, ZerVectImT, phi):
        pi = math.pi
        C = self.compute_C(ZerVectImD, ZerVectImT)
        Aq, Bq = self.comute_Aq_Bq_overq(ZerVectImD, ZerVectImT)
        sum = 0
        for q in range(len(Aq)):
            sum += Aq[q] * math.cos((q + 1) * phi) - Bq[q] * math.sin((q + 1) * phi)
        return C - 4 * pi * sum

    def Overall_derv_func_d(self,ZerVectImD, ZerVectImT, phi):
        pi = math.pi
        Aq, Bq = self.comute_Aq_Bq_overq(ZerVectImD, ZerVectImT)
        sum = 0
        for q in range(len(Aq)):
            sum += q * (Aq[q] * math.sin((q + 1) * phi) + Bq[q] * math.cos((q + 1) * phi))
        return 4 * pi * sum

    def regual_falsi(self,f, x1, x2, ZerVectImD, ZerVectImT, tol=1.0e-6, maxfpos=100):
        xh = 0
        fpos = 0
        if f(ZerVectImD, ZerVectImT, x1) * f(ZerVectImD, ZerVectImT, x2) < 0:
            for fpos in range(1, maxfpos + 1):
                xh = x2 - (x2 - x1) * f(ZerVectImD, ZerVectImT, x2) / (
                            f(ZerVectImD, ZerVectImT, x2) - f(ZerVectImD, ZerVectImT, x1))
                if abs(f(ZerVectImD, ZerVectImT, xh)) < tol:
                    break
                elif f(ZerVectImD, ZerVectImT, x1) * f(ZerVectImD, ZerVectImT, xh) < 0:
                    x2 = xh
                else:
                    x1 = xh
        else:
            print("no root exists")
        return xh, fpos