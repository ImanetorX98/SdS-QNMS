import numpy as np
from sympy import *
from third import *
from nr import *
from SELETTORE_FUNC_SQL import *
from tqdm import tqdm
import mpmath as mp
from rich import print
import os
import re
from tqdm import tqdm
from rosignoli_lib import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import shutil
import traceback

global f
global fx
global df
global ddf
global dddf

global pr
pr = 20

mp.dps = 20

SHOW_PLOTS = os.getenv("QNMS_SHOW_PLOTS", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
if not SHOW_PLOTS:
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass


def show_or_close():
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# Funzione che stampa una riga bianca per aiutare la visualizzazione.
def stampa_riga_bianca():
    terminal_width, _ = shutil.get_terminal_size()
    # Stampa una riga piena con rich
    print("[white on white]" + "=" * terminal_width + "[/]")

"""
Metodo di stima della L di Lipschitz
"""
def lipschitz(df,x0):
    return abs(df(x0))


def to_plain_float(value):
    try:
        return float(value)
    except Exception:
        s = str(value).strip()
        if s.startswith("np.float64(") and s.endswith(")"):
            return float(s[len("np.float64("):-1])
        return float(sympify(value))


def bisection_method(func, a, b, tol=1e-6, max_iterations=100):
    while func(a) * func(b) >= 0:
        a1 = (1 + .1) * a - .1 * b
        b1 = (1 + .1) * b - .1 * a
        a = a1
        b = b1
        #raise ValueError("sign(f(a)) = - sign(f(b))")
        
    iteration = 0
    while (b - a) / 2 > tol and iteration < max_iterations:
        c = (a + b) / 2
        print(f"c{iteration} = {c}")
        if func(c) == 0:
            break
        elif func(c) * func(a) < 0:
            b = c
        else:
            a = c
        iteration += 1
    if iteration >= max_iterations:
        print(f'Max iterations reached')
    return (a + b) / 2 

"""
Metodi di individuazione degli zeri
"""

def newton_raphson_(func, dfunc, x0, tol, max_it, zeros, a, b):
    #print("Using Newton-Raphson method")
    x0 = to_plain_float(x0)
    a = to_plain_float(a)
    b = to_plain_float(b)
    it = 0
    h = (b - a) / 1000
    while it < max_it:
        x0 = to_plain_float(x0)
        funcx = func(x0)
        #print(f'f({x0}) = {funcx} abs = {abs(funcx)}')
        if abs(funcx) <= tol:
            if not any(abs(to_plain_float(zero) - x0) <= tol for zero in zeros):
                zeros.append(to_plain_float(x0))
            x0 += h
            x0 = to_plain_float(check_interval(x0, a, b))
        dfuncx = dfunc(x0)
        if dfuncx == 0:
            break
        x0 += - funcx / check_zero(dfuncx)
        #print(x0)
        x0 = to_plain_float(check_interval(x0, a, b))
        it += 1
        
    return zeros

def halley_(func, dfunc, ddfunc, x0, tol, max_it, zeros, a, b):
    #print("Using Newton-Raphson method")
    x0 = to_plain_float(x0)
    a = to_plain_float(a)
    b = to_plain_float(b)
    it = 0
    h = (b - a) / 1000
    while it < max_it:
        x0 = to_plain_float(x0)
        funcx = func(x0)
        #print(f'f({x0}) = {funcx} abs = {abs(funcx)}')
        if abs(funcx) <= tol:
            if not any(abs(to_plain_float(zero) - x0) <= tol for zero in zeros):
                zeros.append(to_plain_float(x0))
            x0 += h
            x0 = to_plain_float(check_interval(x0, a, b))
        dfuncx = dfunc(x0)
        ddfuncx = ddfunc(x0)
        if dfuncx == 0:
            break
        x0 += - funcx / check_zero(dfuncx - funcx * ddfuncx / (2 * dfuncx))
        #print(x0)
        x0 = to_plain_float(check_interval(x0, a, b))
        it += 1
        
    return zeros

def halley_rescue(func, dfunc, ddfunc, x0, tol, max_it, zeros, a, b):
    #print("Using Newton-Raphson method")
    x0 = to_plain_float(x0)
    a = to_plain_float(a)
    b = to_plain_float(b)
    it = 0
    h = (b - a) / 1000
    while it < max_it:
        x0 = to_plain_float(x0)
        funcx = func(x0)
        #print(f'f({x0}) = {funcx} abs = {abs(funcx)}')
        if abs(funcx) <= tol:
            if not any(abs(to_plain_float(zero) - x0) <= tol for zero in zeros):
                zeros.append(to_plain_float(x0))
            x0 += h
            x0 = to_plain_float(check_interval(x0, a, b))
        dfuncx = dfunc(x0)
        ddfuncx = ddfunc(x0)
        if dfuncx == 0:
            break
        x0 -= funcx / check_zero(dfuncx - funcx * ddfuncx / (2 * dfuncx))
        #print(x0)
        x0 = to_plain_float(check_interval(x0, a, b))
        it += 1
    if len(zeros) == 0:
        zeros.append(bisection_method(func, a, b, 1e-6, 100))
    return zeros

"""
a = ddf(z)/2
b = df(z)
c = f(z)
dz = - c / check( b - c * a / b )
"""

"""
Metodi generalizzati per più punti di innesco
"""

def nr_gen_(my_funct, my_dfunct, x0, tol, max_it, num_zeros, a, b):
    i = 0
    h = (b - a) / 100
    val = True
    zeros = []
    while len(zeros) < num_zeros and val:
        zeros += [num for num in newton_raphson_(my_funct, my_dfunct, x0, tol, max_it, zeros, a, b) if num not in zeros]
        #zeros += [num for num in newton_raphson_interval(my_funct, my_dfunct, a, b, tol, max_it, pr) if num not in zeros]
        x0 += h  # Modifica il punto di innesco per il prossimo zero
        i += 1
        val = (i<100)
    return zeros

def halley_gen_(my_funct, my_dfunct, my_ddfunct, x0, tol, max_it, num_zeros, a, b):
    i = 0
    h = (b - a) / 100
    val = True
    zeros = []
    while len(zeros) < num_zeros and val:
        zeros += [num for num in halley_(my_funct, my_dfunct, my_ddfunct, x0, tol, max_it, zeros, a, b) if num not in zeros]
        #zeros += [num for num in newton_raphson_interval(my_funct, my_dfunct, a, b, tol, max_it, pr) if num not in zeros]
        x0 += h  # Modifica il punto di innesco per il prossimo zero
        i += 1
        val = (i<100)
    return zeros

def halley_gen_rescue(my_funct, my_dfunct, my_ddfunct, x0, tol, max_it, num_zeros, a, b):
    i = 0
    h = (b - a) / 100
    val = True
    zeros = []
    while len(zeros) < num_zeros and val:
        zeros += [num for num in halley_rescue(my_funct, my_dfunct, my_ddfunct, x0, tol, max_it, zeros, a, b) if num not in zeros]
        #zeros += [num for num in newton_raphson_interval(my_funct, my_dfunct, a, b, tol, max_it, pr) if num not in zeros]
        x0 += h  # Modifica il punto di innesco per il prossimo zero
        i += 1
        val = (i<100)
    return zeros

def sel_func(expr = None):
    if expr is None:
        
        f, df, ddf, dddf, fx = seleziona_funzione_from_db()
        return f, df, ddf, dddf, fx   
    try:
        expr = expr.replace('r', 'z')

        z = symbols('z')
        fx = sympify(expr)
        dfx = diff(fx, z)
        ddfx = diff(dfx, z)
        dddfx = diff(ddfx, z)
        
        f = lambdify(z, fx, 'numpy')
        df = lambdify(z, dfx, 'numpy')
        ddf = lambdify(z, ddfx, 'numpy')
        dddf = lambdify(z, dddfx, 'numpy')
        return f, df, ddf, dddf, fx
    except SympifyError:
        print("Espressione non valida.")

def check_zero(z):
    if abs(z) < 10 ** (-20):
        if z < 0:
            z -= 10 ** (-11)
        elif z >= 0:
            z += 10 ** (-11)
    return z

def check_interval(z, a, b):
    if a <= z <= b:
        return z
    else:
        punto_medio = (a + b) / 2
        return punto_medio

def trova_zeri(a, b, tol = 10 ** (-10), method = 'A' ,expr = None):
    f, df, ddf, dddf, fx = sel_func(expr)
    if method == 'A':
        zeros = nr_gen_(f, df, a, tol, 10, 1, a, b)
    else:
        zeros = halley_gen_(f, df, ddf, a, tol, 10, 1, a, b)
    return zeros

def trova_zeri_rescue(a, b, tol = 10 ** (-8), method = 'A' ,expr = None):
    f, df, ddf, dddf, fx = sel_func(expr)
    if method == 'A':
        zeros = nr_gen_(f, df, a, tol, 10, 1, a, b)
    else:
        zeros = halley_gen_rescue(f, df, ddf, a, tol, 10, 1, a, b)
    return zeros

def indice_valore_piu_vicino(lista, valore):
    indice = min(range(len(lista)), key=lambda i: abs(lista[i] - valore))
    return indice

"""
def trova_max_min(funzione_simb, variabile_simb, a, b):
    derivata = 100 * diff(funzione_simb, variabile_simb)
    derivata_lamb = lambdify(variabile_simb, derivata, 'numpy')
    x0 = a
    h = (b - a) / 1000
    for i in range(10000):
        x0 += h * derivata_lamb(x0)
        #print(x0, derivata_lamb(x0))
        if abs(derivata_lamb(x0)) <= 10**(-10):
            return x0, abs(derivata_lamb(x0))
"""

# Funzione dell'individuazione del massimo del potenziale
def maximum_potential(Vder, Vdder, Vddder, l, re, rc, M):
    x0 = to_plain_float(3 * M)
    for i in range(100000):
        c = to_plain_float(Vder(x0, l))
        b = to_plain_float(Vdder(x0, l))
        a = to_plain_float(Vddder(x0, l) / 2)
        x0 = to_plain_float(x0 - c / check_zero(b - c * a / b))
        if abs(to_plain_float(Vder(x0, l))) <= 10**(-15):
            #print(i)
            return to_plain_float(x0), abs(to_plain_float(Vder(x0, l)))

def psi_In_Out(V,f,r0,r0Star,ni,lExp,omega,omega2,A,B):
    # Inizializzo i simboli
    rStar, r, l = symbols('rStar r l')
    x, z, z0, z02, k, pi = symbols('x z z0 z02 k pi', real=True) # Re-inserisci x0
    # Importo le funzioni sympizzandole
    V = sympify(V)
    f = sympify(f)

    # Dichiaro i valori dei simboli
    z = rStar - r0Star
    Q0 = omega2 - V
    Q0l = lambdify((r,l), Q0, 'numpy')
    Q00 = Q0l(r0,lExp)
    V1 = diff(V,r)
    #print(f'V1 = {V1}')
    Q2 = - diff(V1,r) * f ** 2 + V1 * diff(f,r) * f
    Q2l = lambdify((r,l), Q2, 'numpy')
    Q20 = Q2l(r0,lExp)
    #print(f'V = {V}, f = {f},\nQ0 = {Q0}, Q00 = {Q00},\nQ2 = {Q2}, Q20 = {Q20}')
    pi = np.pi
    z02 = - 2 * Q00 / Q20
    z0 = z02 ** .5
    #print(f'z0 = {z0}')
    k = 1 / 2 * Q20
    
    # Soluzioni esterne ai turning points del potenziale
    # z >= z0
    psi1 = (A * exp(I * k ** .5 * z ** 2 / 2) * z ** ni * (4 * k) ** (ni / 4) * exp(- I * pi * ni / 4)
            + B * exp(- I * k ** .5 * z ** 2 / 2) * z ** (- ni - 1) * (4 * k) ** (- (ni + 1) / 4) * exp(- I * pi * (ni + 1) / 4))
    # z <= -z0
    psi2 = ((A + B * (2 * pi) ** .5 * exp(- I * pi * ni / 2) / gamma(ni + 1)) * exp(I * k ** .5 * z ** 2 / 2) * (- z) ** ni * (4 * k) ** (ni / 4)
            * exp(3 * I * pi * ni / 4) + (B * exp(3 * I * pi * (ni + 1) / 2) - A * (2 * pi) ** .5 * exp(I * pi * ni / 2) / gamma(- ni))
            * exp( - I * k ** .5 * z ** 2 / 2) * (- z) ** (- ni - 1) * (4 * k) ** (- (ni + 1) / 4) * exp( - 3 * I * pi * ni / 4))
    
    psiIn = lambdify(rStar, psi2, 'numpy')
    psiOut = lambdify(rStar, psi1, 'numpy')
    
    return psiIn, psiOut

def alpha_n(M,rho1,n,xe,xo):
    return 2*M*(xe-xo)*(n**2+(2*rho1+2)*n+2*rho1+1)

def beta_n(M,l,s,rho1,n,xe,xc,xo):
    return (-2*M*(2*xe-xc-xo)*(n**2+(4*rho1+1)*n+4*rho1**2+2*rho1)
            -l*(l+1)+2*M*xe*(s**2-1))

def gamma_n(M,l,s,rho1,n,xe,xc,xo):
    return (2*M*(xe-xc)*(n**2+4*rho1*n+4*rho1**2-s**2))

def frobenius_coefficients(xe,xc,xo,M,l,omega,s):
    ke = 2*M*(xe-xc)*(xe-xo)
    rho1 = 1j * omega / 2 * ke
    #print(f'rho1 = {rho1}')
    a = [1]
    a0 = 1
    a.append(-beta_n(M,l,s,rho1,0,xe,xc,xo)*a0/alpha_n(M,rho1,0,xe,xo))
    for n in range(2,101):
        a.append(-(a[n-1]*beta_n(M,l,s,rho1,n-1,xe,xc,xo)
                   +a[n-2]*gamma_n(M,l,s,rho1,n-1,xe,xc,xo))/
                 alpha_n(M,rho1,n,xe,xo))
    return a, rho1

def funzione_u(re,rc,ro,M,l,omega,s):
    x = symbols('x')
    xe = 1 / re
    xc = 1 / rc
    xo = 1 / ro
    a, rho1 = frobenius_coefficients(xe,xc,xo,M,l,omega,s)
    # Computing the sum
    t = (x-xe)/(xc-xe)
    powerSum = sum(coeff * np.power(t,i) for i, coeff in enumerate(a, start=1))
    #print(f'powerSum = {powerSum}')
    # Computing the function
    func = np.power((x-xe),(2*rho1))*powerSum + 1
    #print(f'u(x) = {func}')
    funcl = lambdify(x,func,'numpy')
    #print(f'f(xe={xe}) = {funcl(xe)}')
    return funcl, func

def omega_frobenius(re,rc,ro,M,l,n,s):
    xe = 1 / re
    xc = 1 / rc
    xo = 1 / ro
    ke = 2*M*(xe-xc)*(xe-xo)
    prod = ke*re
    print(f'ke*re = {ke*re}')
    b0 = np.sqrt(l*(l+1)-9/4)
    c0 = n + 1/2
    reOmega = prod*b0*(1-2/3*prod)/re
    imOmega = prod*c0*(1-2/3*prod)/re
    omega = reOmega - 1j * imOmega
    omega2 = omega ** 2
    return omega, omega2

"""
#####################################################################
Implementazione dell'algoritmo di Gundlach-Price-Pullin
Per l'equazione [4 * d^2 / dudv + V] * R(u,v) = 0
#####################################################################
"""

def calcola_differenze(lista):
    differenze = []
    for i in range(1, len(lista)):
        differenza = lista[i] - lista[i - 1]
        differenze.append(differenza)
    return differenze

def symmetricRStar(r,rStar):
    r_arr = np.asarray(r, dtype=float).reshape(-1)
    rstar_arr = np.asarray(rStar, dtype=float).reshape(-1)
    n = min(r_arr.size, rstar_arr.size)
    if n < 3:
        raise ValueError(f"Need at least 3 points to build symmetric r* interval, found {n}.")

    r_arr = r_arr[:n]
    rstar_arr = rstar_arr[:n]

    finite_mask = np.isfinite(r_arr) & np.isfinite(rstar_arr)
    r_arr = r_arr[finite_mask]
    rstar_arr = rstar_arr[finite_mask]
    n = r_arr.size
    if n < 3:
        raise ValueError(f"Need at least 3 finite points to build symmetric r* interval, found {n}.")

    # Keep arrays ordered in r* so left/right slices are consistent.
    if np.any(np.diff(rstar_arr) < 0):
        order = np.argsort(rstar_arr)
        rstar_arr = rstar_arr[order]
        r_arr = r_arr[order]

    # u = t - r*, v = t + r*
    # If the interval crosses 0, use r*=0 as center. Otherwise use the midpoint
    # of the interval, since r* is defined up to an additive constant.
    if rstar_arr[0] <= 0 <= rstar_arr[-1]:
        i0 = int(np.argmin(np.abs(rstar_arr)))
    else:
        target = 0.5 * (rstar_arr[0] + rstar_arr[-1])
        i0 = int(np.argmin(np.abs(rstar_arr - target)))
    rStar0 = float(rstar_arr[i0])

    left = i0
    right = n - 1 - i0
    span = min(left, right)

    if span >= 1:
        start = i0 - span
        stop = i0 + span + 1
    else:
        # No symmetric room around i0: pick the closest local window (size >= 3).
        start = max(0, i0 - 1)
        stop = min(n, i0 + 2)
        if stop - start < 3:
            if start == 0:
                stop = min(n, 3)
            else:
                start = max(0, n - 3)
                stop = n

    rStarUV = rstar_arr[start:stop]
    rUV = r_arr[start:stop]
    if rStarUV.size < 3:
        raise ValueError(
            f"Unable to build a valid r* interval for PDE (size={rStarUV.size}, center_index={i0}, n={n})."
        )

    return rStarUV, rUV, rStar0
    
def diamond_borders_construction(r,rStar,spaceTimeK):
    # Come prima cosa costruiamo i bordi del diamante causale
    # Individuo una sotto lista di r* con estremi simmetrici
    rStarUV, rUV, rStar0 = symmetricRStar(r,rStar)
    hStarUV = calcola_differenze(rStarUV)
    hStarM = np.mean(hStarUV)
    # Inizializzo l'array della soluzione QNM
    dim = int((len(rStarUV)+1)/2)
    i0UV = dim - 1
    phiU = np.zeros(dim) # Bordo v = 0 del diamante causale
    phiV = np.zeros(dim) # Primo u = 0 del diamante causale
    # Calcolo l'asse dei tempi
    # A partire dalle differenze e dall'approssimazione dello 0 in r*,
    # computo i valori discreti di u e v
    t = [0]
    v = [rStar0]
    u = [-rStar0]
    avg = 10
    sig = 3
    phiU[:] = 1 / (sig * np.sqrt(2*np.pi)) * np.exp(-(avg)**2/(2*sig**2))
    for i in tqdm(range(1,dim), desc="Diamond Border Progress", leave=False):
        #t.append(t[i-1] + spaceTimeK * hStarUV[i0+i-1])
        t.append(t[2*i-2] + spaceTimeK * hStarM)
        t.append(t[2*i-1] + spaceTimeK * hStarM)
        v.append(t[i]+rStarUV[i0UV+i])
        u.append(t[i]-rStarUV[i0UV-i])
        phiV[i] = 1 / (sig * np.sqrt(2*np.pi)) * np.exp(-(v[i]-avg)**2/(2*sig**2))    
    return phiU, phiV, u, v, t, rStarUV, rUV, hStarUV, i0UV, hStarM   

# Funzione di salvataggio della soluzione della PDE
def salva_soluzione_pde(phi, hStar, M, L, l, spaceTimeK, folder_name='QNMS_PDE_SOL'):
    # Crea la cartella se non esiste
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Trova il prossimo numero disponibile per il progressivo
    num = 1
    while True:
        # Genera il nome del file con il progressivo
        file_name_txt = f'qnm_pde_sol_{M}_{L}_{l}_{hStar}_{spaceTimeK}_{num}.txt'
        file_name_npy = f'qnm_pde_sol_{M}_{L}_{l}_{hStar}_{spaceTimeK}_{num}.npy'
        file_path_txt = os.path.join(folder_name, file_name_txt)
        file_path_npy = os.path.join(folder_name, file_name_npy)

        # Verifica se il file esiste, se sì, incrementa il progressivo
        if not os.path.exists(file_path_txt) and not os.path.exists(file_path_npy):
            break
        num += 1

    # Scrivi la matrice 'phi' nel file testuale (compatibilità retroattiva)
    with open(file_path_txt, 'w') as file:
        for riga in phi:
            file.write(' '.join(map(str, riga)) + '\n')

    # Salva anche il cache binario per letture future molto più rapide
    np.save(file_path_npy, np.asarray(phi, dtype=np.float64))

    print(f'PDE solution saved in {file_path_txt}')
    print(f'PDE binary cache saved in {file_path_npy}')

# In base ai parametri mi trova il file corrispondente
def trova_file_e_estrai_matrice(M, L, l, hStar, spaceTimeK, folder_name='QNMS_PDE_SOL'):
    if not os.path.isdir(folder_name):
        print(f"No preexisting solution found with the specified parameters.")
        return False, None

    escaped_M = re.escape(str(M))
    escaped_L = re.escape(str(L))
    escaped_l = re.escape(str(l))
    escaped_h = re.escape(str(hStar))
    escaped_k = re.escape(str(spaceTimeK))
    pattern_txt = re.compile(
        rf'^qnm_pde_sol_{escaped_M}_{escaped_L}_{escaped_l}_{escaped_h}_{escaped_k}_(\d+)\.txt$'
    )
    pattern_npy = re.compile(
        rf'^qnm_pde_sol_{escaped_M}_{escaped_L}_{escaped_l}_{escaped_h}_{escaped_k}_(\d+)\.npy$'
    )

    # Trova tutti i file che corrispondono al pattern nella cartella
    file_matches = {}
    for filename in os.listdir(folder_name):
        match_npy = pattern_npy.match(filename)
        if match_npy:
            idx = int(match_npy.group(1))
            entry = file_matches.setdefault(idx, {})
            entry["npy"] = filename
            continue
        match_txt = pattern_txt.match(filename)
        if match_txt:
            idx = int(match_txt.group(1))
            entry = file_matches.setdefault(idx, {})
            entry["txt"] = filename

    if not file_matches:
        print(f'No preexisting solution found with the specified parameters.')
        return False, None

    print(f'Found preexisting PDE solution.')

    # Prendi il progressivo minore e preferisci il file binario
    first_idx = min(file_matches.keys())
    chosen = file_matches[first_idx]
    if "npy" in chosen:
        file_path = os.path.join(folder_name, chosen["npy"])
        matrice = np.load(file_path, allow_pickle=False)
        print(f'Loaded cached binary PDE solution: {file_path}')
        return True, np.asarray(matrice)

    file_path = os.path.join(folder_name, chosen["txt"])
    matrice = np.loadtxt(file_path, dtype=float)
    print(f'Loaded text PDE solution: {file_path}')
    cache_path = os.path.splitext(file_path)[0] + ".npy"
    if not os.path.exists(cache_path):
        np.save(cache_path, np.asarray(matrice, dtype=np.float64))
        print(f'Generated binary PDE cache: {cache_path}')
    return True, np.asarray(matrice)

# Funzione di salvataggio della soluzione della PDE
def salva_t_coordinate(t, hStar, M, L, l, spaceTimeK, folder_name='QNMS_PDE_SOL_T'):
    # Crea la cartella se non esiste
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Trova il prossimo numero disponibile per il progressivo
    num = 1
    while True:
        # Genera il nome del file con il progressivo
        file_name_txt = f't_sol_{M}_{L}_{l}_{hStar}_{spaceTimeK}_{num}.txt'
        file_name_npy = f't_sol_{M}_{L}_{l}_{hStar}_{spaceTimeK}_{num}.npy'
        file_path_txt = os.path.join(folder_name, file_name_txt)
        file_path_npy = os.path.join(folder_name, file_name_npy)

        # Verifica se il file esiste, se sì, incrementa il progressivo
        if not os.path.exists(file_path_txt) and not os.path.exists(file_path_npy):
            break
        num += 1

    # Scrivi la lista 't' nel file testuale (compatibilità)
    with open(file_path_txt, 'w') as file:
        for valore in t:
            file.write(f'{valore}\n')

    # Cache binario per caricamenti futuri più veloci
    np.save(file_path_npy, np.asarray(t, dtype=np.float64))
    print(f't coordinate saved in {file_path_txt}')
    print(f't coordinate binary cache saved in {file_path_npy}')

# In base ai parametri mi trova il file corrispondente
def trova_file_e_estrai_t(M, L, l, hStar, spaceTimeK, folder_name='QNMS_PDE_SOL_T'):
    if not os.path.isdir(folder_name):
        print(f'No preexisting solution found with the specified parameters.')
        return False, None

    escaped_M = re.escape(str(M))
    escaped_L = re.escape(str(L))
    escaped_l = re.escape(str(l))
    escaped_h = re.escape(str(hStar))
    escaped_k = re.escape(str(spaceTimeK))
    pattern_txt = re.compile(
        rf'^t_sol_{escaped_M}_{escaped_L}_{escaped_l}_{escaped_h}_{escaped_k}_(\d+)\.txt$'
    )
    pattern_npy = re.compile(
        rf'^t_sol_{escaped_M}_{escaped_L}_{escaped_l}_{escaped_h}_{escaped_k}_(\d+)\.npy$'
    )

    # Trova tutti i file che corrispondono al pattern nella cartella
    file_matches = {}
    for filename in os.listdir(folder_name):
        match_npy = pattern_npy.match(filename)
        if match_npy:
            idx = int(match_npy.group(1))
            entry = file_matches.setdefault(idx, {})
            entry["npy"] = filename
            continue
        match_txt = pattern_txt.match(filename)
        if match_txt:
            idx = int(match_txt.group(1))
            entry = file_matches.setdefault(idx, {})
            entry["txt"] = filename

    if not file_matches:
        print(f'No preexisting solution found with the specified parameters.')
        return False, None

    # Prendi il progressivo minore e preferisci il file binario
    first_idx = min(file_matches.keys())
    chosen = file_matches[first_idx]
    if "npy" in chosen:
        file_path = os.path.join(folder_name, chosen["npy"])
        t = np.load(file_path, allow_pickle=False)
        print(f'Loaded cached binary t coordinate file: {file_path}')
        return True, np.asarray(t)

    file_path = os.path.join(folder_name, chosen["txt"])
    t = np.loadtxt(file_path, dtype=float)
    print(f'Loaded text t coordinate file: {file_path}')
    cache_path = os.path.splitext(file_path)[0] + ".npy"
    if not os.path.exists(cache_path):
        np.save(cache_path, np.asarray(t, dtype=np.float64))
        print(f'Generated binary t cache: {cache_path}')
    return True, np.asarray(t)

# Plotto e salvo i grafi del profilo temporale di psi
def plottaTimePsi(timeProfile, tPlot, PsiGreek, indexDiff,
                  fl, save_folder='QNMS_PDE_TIME_PROFILE',
                  filename1='normPlot', filename2='logPlot',
                  radius = 0):
    filename1 = os.path.join(save_folder, filename1)
    filename2 = os.path.join(save_folder, filename2)

    cifreSignificative = 5
    
    plt.figure()
    plt.plot(tPlot, timeProfile * radius / fl(radius), label='Time profile')
    plt.xlabel('t')
    plt.ylabel(f'{PsiGreek}(t,r={radius:.{cifreSignificative}g})')
    plt.title(f"{PsiGreek} Time profile")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename1)
    print(f'Plot saved in {filename1}.')
    show_or_close()

    plt.figure()
    plt.semilogy(tPlot, abs(timeProfile * radius / fl(radius)), label='Time profile')
    plt.xlabel('t')
    plt.ylabel(f'|{PsiGreek}(t,r={radius:.{cifreSignificative}g})|')
    plt.title(f"{PsiGreek} logarithmic Time profile")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename2)
    print(f'Plot saved in {filename2}.')
    show_or_close()

# Plotto e salvo i grafi del profilo temporale di psi
def plottaSpacePsi(spaceProfile, rStarPlot, rPlot, fl, PsiGreek,
                  save_folder='QNMS_PDE_SPACE_PROFILE',
                  filename1='normPlot', filename2='logPlot', time=0):
    filename1 = os.path.join(save_folder, filename1)
    filename2 = os.path.join(save_folder, filename2)
    
    cifreSignificative = 5
    
    plt.figure()
    plt.plot(rStarPlot, spaceProfile * rPlot / fl(rPlot) , label='Space profile')
    plt.xlabel('r*')
    plt.ylabel(f'{PsiGreek}(t={time:.{cifreSignificative}g},r*)')
    plt.title(f"{PsiGreek} Space profile")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename1)
    print(f'Plot saved in {filename1}.')
    show_or_close()

    plt.figure()
    plt.semilogy(rStarPlot, abs(spaceProfile * rPlot / fl(rPlot)), label='Space profile')
    plt.xlabel('r*')
    plt.ylabel(f'|{PsiGreek}(t={time:.{cifreSignificative}g},r*)|')
    plt.title(f"{PsiGreek} logarithmic Space profile")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename2)
    print(f'Plot saved in {filename1}.')
    show_or_close()

# Costruisco la matrice che devo invertire
def matriceVettore(timeProfile,p,N):
    height = N - p + 1
    X = np.zeros((height, p))
    for j in range(p):
        for i in range(height):
            X[i,j] = timeProfile[p - 1 - j + i]
    x = np.zeros(height)
    for i in range(height):
        x[i] = timeProfile[i + p]
    return X, x

# Risolvo il problema lineare
def leastSquares(X, x):
    # More stable and typically faster than explicit normal-equation inversion.
    coeffs, *_ = np.linalg.lstsq(X, x, rcond=None)
    return -coeffs

# Deduco le frequenze che costituiscono il segnale
def frequenze_p(alpha,p,h, verbose=True):
    z = symbols('z')
    A = sum(z ** (p-m) * alpha[m] for m in range(p+1))
    if verbose:
        print(f'A(z) = {A}')
    solFreq = solve(A, z)
    #print(f'solFreq = {solFreq}')
    omegaFreqRe = np.ones(p)
    omegaFreqIm = np.ones(p)
    for i in range(p):
        numZ = complex(solFreq[i])
        mod = np.abs(numZ)
        arg = np.angle(numZ)
        #print(f'Re({numZ}) = {np.real(numZ)}, Im({numZ}) = {np.imag(numZ)}')
        #print(f'|{numZ}| = {mod}, arg({numZ}) = {arg}')
        omegaFreqRe[i] = - arg / h
        omegaFreqIm[i] = np.log(mod) / h
        #omegaFreq[i] = 1j / h * (np.log(mod) + 1j * arg)
    #print(omegaFreqRe, omegaFreqIm)
    return omegaFreqRe, omegaFreqIm, solFreq


def extract_dominant_mode_from_profile(time_profile, dt, p=6, n_points=None, verbose=False):
    profile = np.asarray(time_profile, dtype=float).reshape(-1)
    finite_mask = np.isfinite(profile)
    profile = profile[finite_mask]
    if profile.size < max(30, 2 * p + 2):
        raise ValueError(
            f"Not enough finite profile points ({profile.size}) for spectral extraction with p={p}."
        )

    available_n = profile.size - 1
    min_n = 2 * p - 1
    if available_n < min_n:
        raise ValueError(
            f"Insufficient profile length {profile.size} for p={p}: require at least {min_n + 1}."
        )

    if n_points is None:
        n_points = int(max(min_n, min(available_n, 0.7 * available_n)))
    n_points = int(max(min_n, min(available_n, n_points)))

    X, x = matriceVettore(profile, p, n_points)
    alpha = np.concatenate((np.array([1.0]), leastSquares(X, x)))
    freqRe, freqIm, solZ = frequenze_p(alpha, p, dt, verbose=verbose)
    omega = freqRe + 1j * freqIm
    zRe, zIm, x0 = matriceZ(profile, solZ, p)
    coeffs = leastSquares(zRe + 1j * zIm, -x0)
    order = np.argsort(np.abs(coeffs))[::-1]

    for idx in order:
        om = omega[idx]
        if np.isfinite(np.real(om)) and np.isfinite(np.imag(om)) and np.imag(om) < 0:
            return om, coeffs[idx], int(idx), omega, coeffs

    idx0 = int(order[0])
    return omega[idx0], coeffs[idx0], idx0, omega, coeffs


def estimate_mode_with_windows(time_profile, t_values, p=6, start_fracs=(0.25, 0.35, 0.45),
                               end_frac=0.95, verbose=False):
    t_arr = np.asarray(t_values, dtype=float).reshape(-1)
    y_arr = np.asarray(time_profile, dtype=float).reshape(-1)
    length = min(t_arr.size, y_arr.size)
    t_arr = t_arr[:length]
    y_arr = y_arr[:length]

    if length < max(40, 2 * p + 2):
        raise ValueError(f"Not enough points for windowed mode estimation: {length}.")

    samples = []
    # Try several windows and reduced model orders to improve robustness.
    p_candidates = sorted({int(max(2, p)), int(max(2, p - 1)), int(max(2, p - 2)), 4, 3, 2}, reverse=True)
    window_pairs = []
    for sf in start_fracs:
        window_pairs.append((float(sf), float(end_frac)))
    window_pairs.extend([
        (0.05, 0.50),
        (0.10, 0.60),
        (0.15, 0.70),
        (0.20, 0.85),
        (0.30, 0.95),
    ])

    for p_try in p_candidates:
        min_window = max(30, 2 * p_try + 2)
        for start_frac, stop_frac in window_pairs:
            if stop_frac <= start_frac:
                continue
            start = int(max(0, min(length - 2, round(start_frac * length))))
            end = int(max(start + 2, min(length, round(stop_frac * length))))
            if end - start < min_window:
                continue
            t_slice = t_arr[start:end]
            y_slice = y_arr[start:end]
            dt_vals = np.diff(t_slice)
            finite_dt = dt_vals[np.isfinite(dt_vals) & (dt_vals != 0)]
            if finite_dt.size == 0:
                continue
            dt = float(np.median(np.abs(finite_dt)))
            if dt <= 0:
                continue
            try:
                om, coeff, idx, _, _ = extract_dominant_mode_from_profile(
                    y_slice, dt, p=p_try, verbose=verbose
                )
                if np.isfinite(np.real(om)) and np.isfinite(np.imag(om)):
                    samples.append(complex(om))
            except Exception:
                continue

    if not samples:
        # Global fallback with minimum model complexity.
        dt_vals = np.diff(t_arr)
        finite_dt = dt_vals[np.isfinite(dt_vals) & (dt_vals != 0)]
        if finite_dt.size > 0:
            dt = float(np.median(np.abs(finite_dt)))
            for p_try in [4, 3, 2]:
                try:
                    om, _, _, _, _ = extract_dominant_mode_from_profile(
                        y_arr, dt, p=p_try, verbose=verbose
                    )
                    if np.isfinite(np.real(om)) and np.isfinite(np.imag(om)):
                        samples.append(complex(om))
                        break
                except Exception:
                    continue
    if not samples:
        raise ValueError("Unable to estimate a dominant mode from any analysis window.")

    samples = np.asarray(samples, dtype=complex)
    omega_mean = complex(np.mean(samples))
    err_re = float(np.std(np.real(samples))) if samples.size > 1 else 0.0
    err_im = float(np.std(np.imag(samples))) if samples.size > 1 else 0.0
    return omega_mean, err_re, err_im, int(samples.size)


def estimate_mode_fft_fallback(time_profile, t_values):
    t_arr = np.asarray(t_values, dtype=float).reshape(-1)
    y_arr = np.asarray(time_profile, dtype=float).reshape(-1)
    n = min(t_arr.size, y_arr.size)
    if n < 20:
        raise ValueError(f"Not enough points for FFT fallback ({n}).")
    t_arr = t_arr[:n]
    y_arr = y_arr[:n]
    finite_mask = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_arr = t_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    if t_arr.size < 20:
        raise ValueError("Not enough finite points for FFT fallback.")

    dt_vals = np.diff(t_arr)
    finite_dt = dt_vals[np.isfinite(dt_vals) & (dt_vals != 0)]
    if finite_dt.size == 0:
        raise ValueError("Unable to compute dt for FFT fallback.")
    dt = float(np.median(np.abs(finite_dt)))
    if dt <= 0:
        raise ValueError("Non-positive dt in FFT fallback.")

    y_center = y_arr - np.mean(y_arr)
    window = np.hanning(y_center.size)
    spec = np.fft.rfft(y_center * window)
    freqs = np.fft.rfftfreq(y_center.size, dt)
    if freqs.size <= 1:
        omega_re = 0.0
    else:
        idx_peak = int(np.argmax(np.abs(spec[1:])) + 1)
        omega_re = float(2 * np.pi * freqs[idx_peak])

    amp = np.abs(y_center)
    amp_max = float(np.max(amp)) if amp.size else 0.0
    valid = amp > max(amp_max * 1e-3, 1e-14)
    if np.count_nonzero(valid) >= 10:
        slope, intercept = np.polyfit(t_arr[valid], np.log(amp[valid]), 1)
        omega_im = float(slope)
    else:
        total_t = float(np.max(t_arr) - np.min(t_arr))
        omega_im = -1.0 / max(total_t, dt)

    if not np.isfinite(omega_im) or omega_im >= 0:
        omega_im = -max(abs(omega_im), 1e-6)

    return complex(omega_re + 1j * omega_im)

# Costruisco la matrice che devo invertire
def matriceZ(timeProfile,zVec,p):
    X = np.ones((p, p))
    Y = np.ones((p, p))
    for j in range(p):
        for i in range(p):
            numZ = complex(zVec[j])
            mod = np.abs(numZ)
            arg = np.angle(numZ)
            numMat = mod ** i * np.exp(1j * i * arg)
            Re = np.real(numMat)
            Im = np.imag(numMat)
            #print(Re,Im)
            X[i,j] = Re
            Y[i,j] = Im
    x = np.zeros(p)
    x = timeProfile[:p]
    return X, Y, x

# Calcolo le frequenze del ring-down e i relativi coefficienti
def frequenzeCoefficienti(timeProfile,hStar,spaceTimeK,omegaGreek):
    pFreq = chiedi_numero_int(f'Select the number of time profile contributes (p) : ')
    nVal = True
    while nVal:
        nFreq = chiedi_numero_int(f'Select the number of intervals that best approximates the ring down (N>={2*pFreq-1}): ')
        nVal = not (nFreq >= 2 * pFreq - 1)
        if nVal:
            print(f'Please check that N>={2*pFreq-1}')
    X, x = matriceVettore(timeProfile,pFreq,nFreq)
    alpha0 = np.array([1])
    alpha1 = leastSquares(X, x)
    alpha = np.concatenate((alpha0,alpha1))
    freqRe, freqIm, solZ = frequenze_p(alpha,pFreq,hStar*spaceTimeK)
    omegaPDE = freqRe + 1j * freqIm
    zRe, zIm, x = matriceZ(timeProfile,solZ,pFreq)
    Z = zRe + 1j * zIm
    hVec = leastSquares(Z,-x)
    print(f'Showing descending coefficients and their frequency')
    indexes = np.argsort(np.abs(hVec))[::-1]
    sumCoeff = sum(np.abs(hVec[i]) for i in range(len(hVec)))
    cS = 10
    for i in indexes:
        omegaIndex = omegaPDE[i]
        cI = hVec[i]
        segno1 = '+' if np.sign(np.imag(omegaIndex)) > 0 else '-'
        segno2 = '+' if np.sign(np.imag(cI)) > 0 else '-'
        print(f'W|C{i}| = {np.abs(cI)/sumCoeff*100:.{cS}g} %, {omegaGreek}({i}) = ({np.real(omegaIndex):.{cS}g} {segno1} {abs(np.imag(omegaIndex)):.{cS}g}*i), C{i} = ({np.real(cI):.{cS}g} {segno2} {abs(np.imag(cI)):.{cS}g}*i)')
    
    return freqRe, freqIm, hVec.real, hVec.imag       

# Mostro il plot 3D e lo salvo
def mostraSalva3d(matrice, frase='grafico_3d'):
    step = max(1, int(max(matrice.shape) / 300))
    mat_view = matrice[::step, ::step]

    # Crea coordinate x e y per la matrice
    x = np.arange(0, mat_view.shape[0])
    y = np.arange(0, mat_view.shape[1])
    X, Y = np.meshgrid(x, y)

    # Crea una figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotta la matrice in 3D
    ax.plot_surface(X, Y, mat_view, cmap='coolwarm')

    # Aggiungi etichette degli assi
    ax.set_xlabel('Asse v')
    ax.set_ylabel('Asse u')
    ax.set_zlabel('R(u,v)')
        
    # Salva il grafico 3D come immagine
    plt.savefig(f'{frase}.png')
    print(f'3D plot saved in {frase}.png')
    
    # Mostra il grafico 3D
    show_or_close()

# Mostro il plot 2D e lo salvo
def mostraSalva2d(matrice, frase='grafico_2d'):
    fig = plt.figure()
    # Plotto la perturbazione spazio-temporale in 2D (u,v)
    plt.imshow(matrice, cmap='coolwarm', aspect='auto')  

    # Aggiungi una barra dei colori per riferimento
    plt.colorbar()

    # Salvo il grafo
    plt.savefig(f'{frase}.png')
    print(f'2D plot saved in {frase}.png')
    
    # Mostra il grafico
    show_or_close()

# Calcolo il profilo temporale
def sliceTimeProfile(indexDiff, phi, t, dim):
    # I extract the time profile
    # I need to gather all the elements
    # on a diagonal and put them into a numpy array
    # timeIndex è un numero compreso tra 0 e C, divido i casi in base
    # al segno di timeIndex - i0.
    indexMod = abs(indexDiff)
    timeRange = dim - indexMod
    idx = np.arange(timeRange)
    if indexDiff > 0:
        timeProfile = phi[indexDiff + idx, idx]
    elif indexDiff < 0:
        timeProfile = phi[idx, indexDiff + idx]
    elif indexDiff == 0:
        timeProfile = phi[idx, idx]
    else:
        timeProfile = np.zeros(timeRange)
    if indexMod == 0:
        tPlot = np.array(t)[::2]
    else:
        tPlot = t[indexMod:-indexMod]
        tPlot = tPlot[::2]
    return timeProfile, tPlot

# Calcolo il profilo spaziale
def sliceSpaceProfile(indexTime, phi, rStar, r, dim):
    # I extract the space profile
    # I need to gather all the elements
    # on a diagonal and put them into a numpy array
    # timeIndex è un numero compreso tra 0 e C, divido i casi in base
    # al segno di spaceIndex - i0.
    phi = np.flipud(phi)
    indexDiff = indexTime-dim
    indexMod = abs(indexDiff)
    spaceRange = dim - indexMod
    idx = np.arange(spaceRange)
    if indexDiff > 0:
        spaceProfile = phi[indexMod + idx, idx]
    elif indexDiff < 0:
        spaceProfile = phi[idx, indexMod + idx]
    elif indexDiff == 0:
        spaceProfile = phi[idx, idx]
    else:
        spaceProfile = np.zeros(spaceRange)
    spaceProfile = spaceProfile[::-1]
    if indexMod == 0:
        rStarPlot = np.array(rStar)[::2]
        rPlot = np.array(r)[::2]
    else:
        rStarPlot = rStar[indexMod:-indexMod]
        rStarPlot = rStarPlot[::2]
        rPlot = r[indexMod:-indexMod]
        rPlot = rPlot[::2]
    return spaceProfile, rStarPlot, rPlot

# SpaceTime show loop
def spaceTimePlot(phi,PsiGreek,frase='parametri'):
    riplotta=chiedi_conferma(f"State whether you'd like to plot {PsiGreek} SpaceTime profile. Y / N : ")
    while riplotta:
        try:
            # Riduzione matrice per ingrandire le piccole oscillazioni
            redMat = chiedi_numero_int_interval(f"What matrix reduction you want to picture? (0 < n < {phi.shape[0]}) : ",
                                                limite_inferiore=float(0), limite_superiore=float(phi.shape[0]))

            # Definisco la matrice di ausilio
            helpMat = phi[redMat:,redMat:]

            # Mostro e salvo i plot
            save_folder1 = 'SpaceTimeProfile/2D'
            save_folder2 = 'SpaceTimeProfile/3D'
            if not os.path.exists(save_folder1):
                os.makedirs(save_folder1)
            if not os.path.exists(save_folder2):
                os.makedirs(save_folder2)
            filename1 = f'2dNormPlot_{frase}_{redMat}'
            filename2 = f'2dLogPlot_{frase}_{redMat}'
            filename3 = f'3dNormPlot_{frase}_{redMat}'
            filename4 = f'3dLogPlot_{frase}_{redMat}'
            filename1 = os.path.join(save_folder1, filename1)
            filename2 = os.path.join(save_folder1, filename2)
            filename3 = os.path.join(save_folder2, filename3)
            filename4 = os.path.join(save_folder2, filename4)
            mostraSalva2d(helpMat, filename1)
            mostraSalva2d(np.log(np.abs(helpMat)), filename2)
            mostraSalva3d(helpMat, filename3)
            mostraSalva3d(np.log(np.abs(helpMat)), filename4)
                
            # White line print
            stampa_riga_bianca()
                
            # Chiedo se vuole riplottare
            riplotta = chiedi_conferma(f"Confirm whether you'd like to plot again {PsiGreek} SpaceTime profile. Y / N : ")
        except Exception as e:
            print(f'Exception \'{e}\' occurred.')
            traceback.print_exc()
            
            # White line print
            stampa_riga_bianca()
                
            riplotta = chiedi_conferma(f"Confirm whether you'd like to plot again {PsiGreek} SpaceTime profile. Y / N : ")        

# Time slice show loop
def timePlot(phi,PsiGreek,omegaGreek,dim,t,rUV,fl,hStar,spaceTimeK,frase='parametri'):
    riplotta = chiedi_conferma(f"State whether you'd like to plot {PsiGreek} time profile. Y / N : ")
    while riplotta:
        try:
            indexDiff = chiedi_numero_int_interval(f"Which UV matrix diagonal do you pick with respect to the principal one?\n#### 0 - principal\n#### {dim} > n > 0 - positive r*\n#### {-dim} < n < 0 - negative r*\nYour pick : ",
                                                   limite_inferiore=float(f"{-dim}"), limite_superiore=float(f"{dim}"))
            # Mi calcola il profilo temporale
            timeProfile, tPlot = sliceTimeProfile(indexDiff, phi, t, dim)
            #timeProfile[0] = .001
            save_folder = 'QNMS_PDE_TIME_PROFILE'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            filename1 = f'plot_{frase}_{indexDiff}.png'
            filename2 = f'semilogy_{frase}_{indexDiff}.png'
            plottaTimePsi(timeProfile, tPlot, PsiGreek, indexDiff,
                          fl, filename1=filename1, filename2=filename2,
                          radius=rUV[indexDiff+dim])

            # Deduco le frequenze che costituiscono il segnale.
            # Provo con p = 3
            print(f'Len({PsiGreek}) = {len(timeProfile)}')
            freqRe, freqIm, hVecRe, hVecIm = frequenzeCoefficienti(timeProfile,hStar,spaceTimeK,omegaGreek)
            omegaPDE = freqRe + 1j * freqIm
            hVec = hVecRe + 1j * hVecIm

            # White line print
            stampa_riga_bianca()
                
            riplotta = chiedi_conferma(f"Confirm whether you'd like to plot again {PsiGreek} time profile. Y / N : ")
        except Exception as e:
            print(f'Exception \'{e}\' occurred.')

            # White line print
            stampa_riga_bianca()
                
            riplotta = chiedi_conferma(f"Confirm whether you'd like to plot again {PsiGreek} time profile. Y / N : ")

# Time slice show loop
def spacePlot(phi,PsiGreek,omegaGreek,dim,t,rStarUV,rUV,fl,hStar,spaceTimeK,frase='parametri'):
    riplotta = chiedi_conferma(f"State whether you'd like to plot {PsiGreek} space profile. Y / N : ")
    while riplotta:
        try:
            indexTime = chiedi_numero_int_interval(f"Which time instant do you pick?\n#### 0 < t < {2*dim}\nYour pick : ",
                                                   limite_inferiore=float(0), limite_superiore=float(f"{2*dim}"))
                
            # Mi calcola il profilo temporale
            spaceProfile, rStarPlot, rPlot = sliceSpaceProfile(indexTime, phi, rStarUV, rUV, dim)
            #timeProfile[0] = .001
            save_folder = 'QNMS_PDE_SPACE_PROFILE'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            filename1 = f'plot_{frase}_{indexTime}.png'
            filename2 = f'semilogy_{frase}_{indexTime}.png'
            plottaSpacePsi(spaceProfile, rStarPlot, rPlot, fl, PsiGreek,
                           filename1=filename1, filename2=filename2,
                           time=t[indexTime])
            """
            # Deduco le frequenze che costituiscono il segnale.
            # Provo con p = 3
            print(f'Len({PsiGreek}) = {len(spaceProfile)}')
            freqRe, freqIm, hVecRe, hVecIm = frequenzeCoefficienti(spaceProfile,hStar,spaceTimeK,omegaGreek)
            omegaPDE = freqRe + 1j * freqIm
            hVec = hVecRe + 1j * hVecIm
            """
            # White line print
            stampa_riga_bianca()
                
            riplotta = chiedi_conferma(f"Confirm whether you'd like to plot again {PsiGreek} space profile. Y / N : ")
        except Exception as e:
            print(f'Exception \'{e}\' occurred.')

            # White line print
            stampa_riga_bianca()
                
            riplotta = chiedi_conferma(f"Confirm whether you'd like to plot again {PsiGreek} space profile. Y / N : ")
    
# Con gli r calcolo V(r), con V(r) e rStar
# interpolo attorno al massimo di V e computo le derivate
def interpolatePotential(V0eff,lExp,V1eff, V2eff, V3eff, re, rc, M, r, rStar, margin=3):
    r0, defect = maximum_potential(V1eff, V2eff, V3eff, lExp, re, rc, M)
    r0 = to_plain_float(r0)
    r_arr = np.asarray(r, dtype=float).reshape(-1)
    rstar_arr = np.asarray(rStar, dtype=float).reshape(-1)
    if r_arr.size != rstar_arr.size:
        n = min(r_arr.size, rstar_arr.size)
        r_arr = r_arr[:n]
        rstar_arr = rstar_arr[:n]
    indice = indice_valore_piu_vicino(r_arr, r0)
    rStar0 = to_plain_float(rstar_arr[indice])
    margin = int(max(2, margin))
    start = max(0, indice - margin)
    stop = min(len(r_arr), indice + margin + 1)
    rInt = r_arr[start:stop]
    rStarInt = rstar_arr[start:stop]
    if len(rInt) < 7:
        raise ValueError(
            f"Interpolation window too small ({len(rInt)} points) around r0={r0} with margin={margin}; need >= 7."
        )

    vInt = np.asarray([to_plain_float(V0eff(radius, lExp)) for radius in rInt], dtype=float)
    deg = min(6, len(rStarInt) - 1)
    if deg < 6:
        raise ValueError(f"Interpolation degree {deg} is insufficient for sixth derivative.")

    coeffs = np.polyfit(rStarInt, vInt, deg=deg)
    poly = np.poly1d(coeffs)
    V0 = to_plain_float(poly(rStar0))
    V2 = to_plain_float(np.polyder(poly, 2)(rStar0))
    V3 = to_plain_float(np.polyder(poly, 3)(rStar0))
    V4 = to_plain_float(np.polyder(poly, 4)(rStar0))
    V5 = to_plain_float(np.polyder(poly, 5)(rStar0))
    V6 = to_plain_float(np.polyder(poly, 6)(rStar0))

    return V0, V2, V3, V4, V5, V6
