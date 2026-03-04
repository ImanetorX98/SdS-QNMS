import math as m
import numpy as np
from sympy import *
from third import *

#y = symbols('y')
pr = 20

def newton_raphson(func, dfunc, x0, tol, max_it, zeros):
    #print("Using Newton-Raphson method")
    it = 0
    while it < max_it:
        fx = func(x0)
        if abs(fx) <= tol:
            if not any(abs(zero-x0) <= tol for zero in zeros):
                zeros.append(N(x0,pr))
            x0 += .5
        dfx = dfunc(x0)
        if dfx == 0:
            break
        x0 += - fx / dfx
        it += 1
    return zeros

def rosign_third(func, dfunc, ddfunc, dddfunc, x0, tol, max_it, zeros):
    n = 0
    while n < max_it:
        a0 = func(x0)
        a1 = dfunc(x0)
        a2 = ddfunc(x0)
        a3 = dddfunc(x0)
        A = 3*a2/a3
        B = 6*a1/a3
        C = 6*a0/a3
        delta = solve_t(1,A,B,C)[0]
        if abs(delta) <= tol:
            if not any(abs(zero-x0) <= tol for zero in zeros):
                zeros.append(N(x0,pr))
            x0 += .5
        x0 += delta
        n += 1
    return zeros

def rosign_third_1(func, dfunc, ddfunc, dddfunc, x0, tol, max_it, zeros):
    n = 0
    while n < max_it:
        a0 = func(x0)
        a1 = dfunc(x0)
        a2 = ddfunc(x0)
        a3 = dddfunc(x0)
        A = 3*a2/a3
        B = 6*a1/a3
        C = 6*a0/a3
        delta = solve_t(1,A,B,C)[1]
        if abs(delta) <= tol:
            if not any(abs(zero-x0) <= tol for zero in zeros):
                zeros.append(N(x0,pr))
            x0 += .5
        x0 += delta
        n += 1
    return zeros

def rosign_third_2(func, dfunc, ddfunc, dddfunc, x0, tol, max_it, zeros):
    n = 0
    while n < max_it:
        a0 = func(x0)
        a1 = dfunc(x0)
        a2 = ddfunc(x0)
        a3 = dddfunc(x0)
        A = 3*a2/a3
        B = 6*a1/a3
        C = 6*a0/a3
        delta = solve_t(1,A,B,C)[2]
        if abs(delta) <= tol:
            if not any(abs(zero-x0) <= tol for zero in zeros):
                zeros.append(N(x0,pr))
            x0 += .5
        x0 += delta
        n += 1
    return zeros

# Esempio di funzione per cui trovare gli zeri
def my_f(x):
    return x**3/6-x**2/2-x+1

# Derivata della funzione
def my_df(x):
    return x**2/2-x-1

# Parametri per il metodo di Newton-Raphson
x0 = -10 # Punto di partenza per il primo zero
tolerance = 1e-10 # Tolleranza per l'approssimazione dello zero
max_iterations = 100 # Numero massimo di iterazioni per ogni zero
num_zeros = 10 # Numero di zeri da individuare

# Metodo di N-R per gruppi di zeri
def nr_gen(my_funct, my_dfunct, x0, tol, max_it, num_zeros):
    i = 0
    val = True
    zeros = []
    while len(zeros) < num_zeros and val:
        zeros += [num for num in newton_raphson(my_funct, my_dfunct, x0, tol, max_it, zeros) if num not in zeros]
        x0 += .1  # Modifica il punto di innesco per il prossimo zero
        #print(x0)
        i += 1
        val = (i<100)
    return zeros

def rosign_gen(func, dfunc, ddfunc, dddfunc, x0, tol, max_it, num_zeros):
    i = 0
    val = True
    zeros0 = []
    zeros1 = []
    zeros2 = []
    zeros = []
    while len(zeros) < num_zeros and val:
        zeros0 += [num for num in rosign_third(func, dfunc, ddfunc, dddfunc, x0, tol, max_it, zeros) if num not in zeros]
        zeros1 += [num for num in rosign_third_1(func, dfunc, ddfunc, dddfunc, x0, tol, max_it, zeros) if num not in zeros]
        zeros2 += [num for num in rosign_third_2(func, dfunc, ddfunc, dddfunc, x0, tol, max_it, zeros) if num not in zeros]
        zeros += [num for num in zeros0 if num not in zeros]
        zeros += [num for num in zeros1 if num not in zeros]
        zeros += [num for num in zeros2 if num not in zeros]
        x0 += .1  # Modifica il punto di innesco per il prossimo zero
        #print(x0)
        i += 1
        val = (i<100)
        #print(zeros, x0)
    return zeros

#zeros = nr_gen(my_f, my_df, x0, tolerance, max_iterations, num_zeros)
#print("Gli zeri individuati sono:", zeros)


