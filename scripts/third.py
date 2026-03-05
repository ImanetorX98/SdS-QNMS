import numpy as np
import math as m
from sympy import *
"""
from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
from sympy.tensor.tensor import tensor_indices
"""
x, y, z = symbols('x,y,z')
init_printing(use_unicode=False, wrap_line=False)

def solve_t(a,b,c,d):
    p = c/a - (b**2)/(3*a**2)
    q = d/a - (b*c)/(3*a**2) + (2*b**3)/(27*a**3)
    delta = p**3/27 + q**2/4
    try: # Se ci troviamo nel campo reale utilizza i tre metodi disponibili
        if delta > 0:
            d = np.sqrt(delta)
            u = np.cbrt( - q/2 + d)
            v = np.cbrt( - q/2 - d)
            x1 = u + v - b/(3*a)
            x2 = u*(-1+(-1)**(1/2)*np.sqrt(3))/2 + v*(-1-(-1)**(1/2)*np.sqrt(3))/2 - b/(3*a)
            x3 = u*(-1-(-1)**(1/2)*np.sqrt(3))/2 + v*(-1+(-1)**(1/2)*np.sqrt(3))/2 - b/(3*a)
        elif delta < 0:
            d = np.sqrt(-delta)
            rho = 2 * (-p/3)**(1/2)
            theta = m.atan(d/(-q/2))
            x1 = rho*np.cos(theta/3+0*np.pi/3) - b/(3*a)
            x2 = rho*np.cos(theta/3+2*np.pi/3) - b/(3*a)
            x3 = rho*np.cos(theta/3+4*np.pi/3) - b/(3*a)
        elif delta == 0:
            x1 = -2*(q/2)**(1/3) - b/(3*a)
            x2 = (q/2)**(1/3) - b/(3*a)
            x3 = x2
    except: # Se ci troviamo nel campo complesso utilizza il metodo unico
        d = (delta)**(1/2)
        u = ( - q/2 + d)**(1/3)
        v = ( - q/2 - d)**(1/3)
        x1 = u + v - b/(3*a)
        x2 = u*(-1+(-1)**(1/2)*np.sqrt(3))/2 + v*(-1-(-1)**(1/2)*np.sqrt(3))/2 - b/(3*a)
        x3 = u*(-1-(-1)**(1/2)*np.sqrt(3))/2 + v*(-1+(-1)**(1/2)*np.sqrt(3))/2 - b/(3*a)    

    if x1 is not None:
        return x1, x2, x3
    else:
        print(f"Computation error. p = {p}, q = {q}")

