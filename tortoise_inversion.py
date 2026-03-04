from sympy import *
from pypiexperiment import rosignoli_lib as rl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from qnms_utils import *
import subprocess
import signal
import os
import unicodedata
import curses
from rich import print
import shutil
import time
import re as REX
from rosignoli_lib import *

SHOW_PLOTS = os.getenv("QNMS_SHOW_PLOTS", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
if not SHOW_PLOTS:
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

_PLOT_COUNTERS = {}


def show_or_close():
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def next_progressive_plot_path(save_folder, pattern, stem):
    key = (save_folder, pattern.pattern)
    if key not in _PLOT_COUNTERS:
        max_progressive = 0
        for filename in os.listdir(save_folder):
            match = pattern.match(filename)
            if match:
                max_progressive = max(max_progressive, int(match.group(1)))
        _PLOT_COUNTERS[key] = max_progressive
    _PLOT_COUNTERS[key] += 1
    return os.path.join(save_folder, f"{stem}{_PLOT_COUNTERS[key]}.png")

"""
############################################################################
Funzioni utili all'inversione di coordinate
############################################################################
"""

def calcola_differenze(lista):
    differenze = []
    for i in range(1, len(lista)):
        differenza = lista[i] - lista[i - 1]
        differenze.append(differenza)
    return differenze

"""
############################################################################
Funzioni per il salvataggio dei dati dell'inversione
############################################################################
"""

def calcola_momenti_statistici_e_salva(h, hstareff, num):
    # num vale 1 per rootfinding, 2 per I Taylor e 3 per II Taylor
    print(f'### Hypothetical h* : {h}')
        
    # Calcolo della media utilizzando np.mean()
    media = abs(np.mean(hstareff))
    print("### <h*> pullback :", media)

    # Calcolo della varianza utilizzando np.var()
    varianza = np.var(hstareff)
    print(f"### {sigma}* pullback :", np.sqrt(varianza))

    # Calcolo del difetto di media utilizzando 2*(B-A)/(B+A)
    difetto = 200 * abs((h - media) / (h + media))
    print(f"### Defect h* pullback : {difetto} %")

    # Segno i risultati dell'inversione di coordinate con I ordine
    file_name = 'qnms_tortoise_inversion_results_.txt'
    with open(file_name, 'a') as file:
        file.write(f'{num} {M} {L} {hrad} {hstar} {lettera} {h} {media} {varianza} {difetto} {elapsed_time} {tol}\n')
    
    print(f"Momenti statistici salvati sul file {file_name}")

def salva_output_inversione_testuggine(r_vals, rstar_pullback_vals, num, folder_name = 'TORTOISE_INVERSION_OUTPUT'):
    # Crea la cartella se non esiste
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Scrivere le liste in un file .txt
    file_name = f'output_{num}_{M}_{L}_{hrad}_{hstar}_.txt'
    file_path = os.path.join(folder_name, file_name)  # Percorso completo del file

    if num == 1:
        with open(file_path, 'w') as file:
            for i in range(len(r_vals)):
                file.write(f'{r_vals[i]} {rstar_pullback_vals[i][0]}\n')
    else:
        # Salva i dati su un file di testo in una cartella
        with open(file_path, 'w') as file:
            for i in range(len(r_vals)):
                file.write(f'{r_vals[i]} {rstar_pullback_vals[i]}\n')
            
    if num == 1:
        print(f'Root finding & interpolation data saved in {file_path}')
    elif num == 2:
        print(f'I order data saved in {file_path}')
    elif num == 3:
        print(f'II order data saved in {file_path}')

# Funzione che stampa una riga bianca per aiutare la visualizzazione.
def stampa_riga_bianca():
    terminal_width, _ = shutil.get_terminal_size()
    # Stampa una riga piena con rich
    print("[white on white]" + "=" * terminal_width + "[/]")

def filtra_punti(r_vals, rstar_vals, rstar_pullback_vals):
    # Verifica se il numero totale di punti supera max_points
    if len(r_vals) > max_points:
        step = len(r_vals) // max_points  # Trova il passo da utilizzare
        sampled_r_vals = r_vals[::step]
        sampled_rstar_vals = rstar_vals[::step]
        sampled_rstar_pullback_vals = rstar_pullback_vals[::step]
    else:
        sampled_r_vals = r_vals
        sampled_rstar_vals = rstar_vals
        sampled_rstar_pullback_vals = rstar_pullback_vals
    return sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals

def compila_valori_taylor(rstar_vals, r_vals, rstar_pullback_vals, hstarnum, increment):
    # Ricavo i valori dei raggi* iterativamente
    for i in range(hstarnum):
        rstar_vals.append(rstar_vals[-1] + hstar)
            
    # Ricavo i valori dei raggi iterativamente
    for i in tqdm(range(hstarnum)):
        r_vals.append(r_vals[-1] + increment(r_vals[-1]))
            
    # Dai raggi calcolati calcolo i raggi*
    for i in range(hstarnum):
        rstar_pullback_vals.append(tl(r_vals[i+1]))

    return rstar_vals, r_vals, rstar_pullback_vals

def save_plot_with_progressive(x, y, ypullback, labelx='r', labely='r*', frase='r vs r*', save_folder='TORTOISE_INVERSION_PLOTS'):
    # Creazione del grafico
    plt.figure()
    plt.scatter(x, y, label='Hypothetical r*')
    plt.scatter(x, ypullback, label='Effective r*') if ypullback else None
    plt.axvline(x=re, color='red', linestyle='--', label='re')  # Linea verticale per re
    plt.axvline(x=rc, color='blue', linestyle='--', label='rc')  # Linea verticale per ec
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(frase)
    plt.legend()
    plt.grid(True)

    # Crea la cartella se non esiste
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    pattern = r'rstar_' + REX.escape(word) + r'_(\d+)\.png$'
    filename = next_progressive_plot_path(save_folder, REX.compile(pattern), f"rstar_{word}_")

    # Salvataggio del plot
    plt.savefig(filename)
    print(f'Plot saved in {filename}')

    # Mostra il plot
    show_or_close()

# Il main program
if __name__ == "__main__":
    try:
        # Dichiaro le variabili globali
        global M
        global L
        global hrad
        global hstar
        global lettera
        global tol
        global word
        global re
        global rc
        
        # Ottieni la larghezza della finestra terminale
        terminal_width, _ = shutil.get_terminal_size()
        
        # Stampa una riga piena con rich
        print("[white on white]" + "=" * terminal_width + "[/]")
        
        # Symbols' definition
        w, t, r, V, M, A, B, C, L = symbols('w t r V M A B C L', real = True)
        l, s = symbols('l s', integer = True)
        re, rc, ro, rstar = symbols('re rc ro rstar', real = True)
        ke, kc, ko = symbols('ke kc ko', real = True)
        wstar, wrad = symbols('wstar wrad', real = True)
        funcrad = Function('funcrad')(r)
        
        # First we define the metric parameters
        global sigma
        ignore_warnings()
        Lambda = unicodedata.lookup('GREEK CAPITAL LETTER LAMDA')
        sigma = unicodedata.lookup('GREEK SMALL LETTER SIGMA')
        print(f'Type SdS metric\'s parameters values')
        M = rl.chiedi_numero(f"BH Mass M = ")
        correctLamb = False
        while not correctLamb:
            L = 10 ** - ( rl.chiedi_numero_pos(f"Cosmological constant {Lambda} = 10 ** -"))
            cosPar = 9 * M ** 2 * L
            correctLamb = (cosPar < 1)
            if not correctLamb:
                print(f'{Lambda}*M**2 needs to be < 1 / 9')
                print(f'Suggested {Lambda} value < {1 / (9 * M ** 2)}')
        
        # Checking if root-finding is needed
        root_finding = chiedi_conferma("Do you want to apply root finding? S - yes or N - no : ")
        
        # Then we define h and h*
        hrad = 10 ** - ( rl.chiedi_numero_pos("Direct h = 10 ** -"))
        hstar = 10 ** - ( rl.chiedi_numero_pos("Tortoise h* = 10 ** -"))

        # I write on the file wether I apply root_finding or not.
        lettera = 'S' if root_finding else 'N'

        # We initialize the metric grr**-1 function
        f = 1 - 2 * M / r - L / 3 * r ** 2
        fl = lambdify(r, f, 'numpy')
        finvl = lambdify(r, f**(-1), 'numpy')
        equation = Eq(f, 0)
        
        # Solve the equation analitically
        print('Solving analitically g_rr ** -1 = 0')
        solutions = solve(f, r)
        q = 6 * M / L
        p = - 3 / L
        
        fs = r**3 + p * r + q
        print(f'Associated horizon equation : {fs}')
        solutions = solve(fs, r)

        
        print("Numerical solutions:")
        # rhorizon = r3, rcosmological = r1, rneg = r2
        r1 = float(2 / np.sqrt(L) * np.cos(np.arctan2(np.sqrt((cosPar) ** (-1) - 1), -1) / 3))
        print(f'Cosmological radius  rc = {r1}')

        r2 = float(2 / np.sqrt(L) * np.cos(np.arctan2(np.sqrt((cosPar) ** (-1) - 1), -1) / 3 + 2 * np.pi / 3))
        print(f'Negative radius      ro = {r2}')

        r3 = float(2 / np.sqrt(L) * np.cos(np.arctan2(np.sqrt((cosPar) ** (-1) - 1), -1) / 3 + 4 * np.pi / 3))
        print(f'BH Horizon           re = {r3}')

        # We check wheter we solved the equation correctly
        ft = (r - r1) * (r - r2) * (r - r3) / r * ( - L / 3)
        ft = expand(ft)
        print(f'Numerically produced equation : (r - re) * (r - rc) * (r - ro) / r * ( - {Lambda} / 3) = {ft}')
        print(f'Original equation : {f}')

        ri = [
            float(2 / np.sqrt(L) * np.cos(np.arctan2(np.sqrt((cosPar) ** (-1) - 1), -1) / 3 + 2 * i * np.pi / 3))
            for i in range(3)
        ]
        fdiff = diff(f, r)
        fDiff = lambdify(r, fdiff, 'numpy')
        fddiff = diff(fdiff, r)
        fDDiff = lambdify(r, fddiff, 'numpy')
        
        ki = [float(abs(fDiff(radius)) / 2) for radius in ri]
        print(f'Associated surface gravities:')
        print(ki)

        # Analytical solution to the integral
        t = .5 * ( log(r / ri[2] - 1) / ki[2] - log(1 - r / ri[0]) / ki[0] + log(1 - r / ri[1]) / ki[1] )
        tprint = ( log(r / re - 1) / (2 * ke) - log(1 - r / rc) / (2 * kc) + log(1 - r / ro) / (2 * ko))
        tl = lambdify(r, t, 'numpy')
        print(f'Tortoise coordinates function definition : ')
        print(f'r*(r) = {tprint}')
        print(f'r*(r) = {t}')
        rc, ro, re = ri
        kc, ko, ke = ki

        # Scrivo i valori dei parametri su un file
        with open(f'qnms_parameters_.txt', 'w') as file:
            file.write(f'{M}\n{L}\n{hrad}\n{hstar}\n{lettera}\n{re}\n{rc}\n{ke}')

        # Let's ask the user what derivative he wants to exteem
        minDer = 1/(1 - np.cbrt(cosPar))
        derStar = chiedi_numero_interval(f"Define a derivative value greater then {minDer} : ", minDer)
        derComp = 1 - 1 / derStar
        
        # Let's get the extremal points 
        rderRc = 2 * np.sqrt(derComp / L) * np.cos(np.arctan2(np.sqrt(derComp ** 3 / cosPar - 1), - 1) / 3)
        print(f'The radius close to rc at which tortoise coordinates assume derivative {derStar} is r = {rderRc}')
        
        rderRe = 2 * np.sqrt(derComp / L) * np.cos(np.arctan2(np.sqrt(derComp ** 3 / cosPar - 1), - 1) / 3 + 4 * np.pi / 3)
        print(f'The radius close to re at which tortoise coordinates assume derivative {derStar} is r = {rderRe}')
        
        # Setting r, r* extremal values
        print(f'Determining the simulation\'s extremal points.')
        global rmin
        global rmax
        global rstarmin
        global rstarmax
        global max_points
        
        max_points = int(1e5)
        tol = 10 ** (-2) * rl.chiedi_numero_pos("What horizon proximity percentual tolerance do you need? ")
        ratio = chiedi_numero_int_pos("In what ratio you want the new dominion extrema defects to be in? ") ** (-1)
        rmin = (re * (ratio + 1 - ratio * tol) + rc * ratio * tol) / (ratio + 1)
        rmax = (rc * (ratio + 1 - tol) + re * tol) / (ratio + 1)
        rstarmin = float(tl(rmin))
        rstarmax = float(tl(rmax))

        

        # Dichiaro la stringa con la quale salvo i plots
        word = f'{M}_{L}_{hrad}_{hstar}_{tol}'
        
        print(f'Expressing min-max values.')
        print(f'### r*(rmin) = rmin* ===> r*({rmin}) = {rstarmin}')
        print(f'### r*(rmax) = rmax* ===> r*({rmax}) = {rstarmax}')
        
        print(f'### (rmin - re) / (rc - re) = {(rmin - re) / (rc - re) * 100} %')
        print(f'### (rc - rmax) / (rc - re) = {(rc - rmax) / (rc - re) * 100} %')
        print(f'### (right defect) / (left defect) = {(rc - rmax) / (rmin - re)}')
        # Plotting the function graph
        print(f'Plotting the function\'s graph.')
        hnum = int((rmax - rmin) / hstar)
        r_vals = np.linspace(rmin, rmax, max_points)
        rstar_vals = tl(r_vals)
        # Grafico della funzione r*(r)
        
        #fig, ax = plt.subplots(figsize=(10, 7))
        plt.scatter(r_vals, rstar_vals, label= f'r*(r)')
        plt.axvline(x=re, color='red', linestyle='--', label='re')  # Linea verticale per re
        plt.axvline(x=rc, color='blue', linestyle='--', label='rc')  # Linea verticale per ec
        plt.xlabel('r')
        plt.ylabel('r*')
        plt.title('Directly computed r*(r) : r vs r*')
        plt.legend()
        fileName = f'tortoisePlot_{M}_{L}_{tol}_{ratio}_.png'
        os.makedirs("TORTOISE_INVERSION_PLOTS", exist_ok=True)
        plt.savefig(f'TORTOISE_INVERSION_PLOTS/{fileName}')
        print(f'Plot saved in {fileName}')
        show_or_close()
        
        # Stampa riga bianca per tutta la larghezza della pagina
        stampa_riga_bianca()
        
        """
        ############################################################################
        Function inversion with evenly distributed r*, root finding and interpolation
        ############################################################################
        """
        if root_finding:
            print(f'Function inversion with evenly distributed r*, root finding and interpolation')
            zeros = [[float(rmin)]]
            radii = [float(rstarmin)]
            hnum = int((rc - re) / hrad)
            A = float(rstarmin)
            B = float(rstarmax)
            method = rl.chiedi_char("""Select your favourite root-finding method:
A - Newton Raphson (m = 2)
B - Halley (m = 3)
Choice : """)
            
            lipschitzConstMin = lipschitz(finvl, A)
            lipschitzConstMid = lipschitz(finvl, (A + B) / 2)
            lipschitzConstMax = lipschitz(finvl, B)
            
            # Inverto la funzione con root finding enhanced
            start_time = time.time()
            
            for i in tqdm(range(int((B - A) / hrad) - 1)):
                radius = float(A + (i + 1) * hrad)
                radii.append(radius)
                zero = trova_zeri(A, B, 10**(-10), method, str(t - radius))
                if len(zero) == 0: # Rescue Attempt in case of emergency
                    k = float((radius - A) / (B - A))
                    a = float((1 - (k - .005)) * rmin + (k - .005) * rmax)
                    b = float((1 - (k + .005)) * rmin + (k + .005) * rmax)
                    lin = np.linspace( a, b, 100)
                    lipschitzConst = finvl(zeros[0][0])
                    tempfunc = lambdify(r, exp((t - radius) / 100) - 1, 'numpy')
                    vals = numeri_piu_vicini_a_zero(tempfunc, lin)
                    zero = trova_zeri(float(vals[1]), float(vals[3]), 10**(-10), method, str(exp((t - radius) / 100) - 1))
                zeros.append(zero)
                
            #print(zeros)
            radii.append(float(rstarmax))
            zeros.append([float(rmax)])
            zeros = fill_list_with_interpolation(zeros)

            # Comincia a misurare il tempo necessario alla generazione dei punti
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f'We proceed verifying that the interpolated list provides constant h*')
            
            # Dai raggi calcolati calcolo i raggi*
            rstarpullback = [rstarmin]
            for i in range(len(zeros) - 1):
                try:
                    num = zeros[i+1][0]
                    #print(tl(complex(num)))
                    rstarpullback.append(np.real(tl(complex(num))))
                except Exception as e:
                    print(e)
                    
            #print(f'{len(rstarpullback)} {rstarpullback}')
            # E' un pezzo di codice che si ripete tre volte, posso automatizzarlo in un unico metodo
            
            # Calcolo gli scarti tra raggi* adiacenti
            hstareff = calcola_differenze(rstarpullback)
            
            # Calcolo i momenti statistici degli r* pullback
            calcola_momenti_statistici_e_salva(hrad, hstareff, 1)

            # Filtra i valori fino ad un massimo di max_points
            sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals = filtra_punti(zeros, radii, rstarpullback)

            # Plotto i punti e salvo il grafico
            save_plot_with_progressive(sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals, 'r', 'r*', frase='Inversion with root finding and interpolation: r vs r*')

            # Salvo gli output dell'inversione
            salva_output_inversione_testuggine(radii, zeros, 1)
            
            # Stampa riga bianca per tutta la larghezza della pagina
            stampa_riga_bianca()

        """
        #####################################################################
        Function inversion with distribution of r based on Taylor's truncation
        Primo ordine
        #####################################################################
        """
        print(f'I order inversion computation')
        # Priviamo ad utilizzare il primo termine del polinomio di Taylor
        increment = lambdify(r, hstar * f, 'numpy')
        r_vals = [rmin]
        rstar_vals = [rstarmin]
        rstar_pullback_vals = [rstarmin]
        hstarnum = int((rstarmax - rstarmin) / hstar)
        start_time = time.time()
        
        # Ricavo i valori di: r*, r ed r*_pullback
        rstar_vals, r_vals, rstar_pullback_vals = compila_valori_taylor(rstar_vals, r_vals, rstar_pullback_vals, hstarnum, increment)
        
        # Comincia a misuare il tempo necessario alla generazione delle coordinate
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Filtra i valori fino ad un massimo di max_points
        sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals = filtra_punti(r_vals, rstar_vals, rstar_pullback_vals)

        # Calcolo gli scarti tra raggi* adiacenti
        hstareff = calcola_differenze(rstar_pullback_vals)
        
        # Calcolo i momenti statistici degli r* pullback
        calcola_momenti_statistici_e_salva(hstar, hstareff, 2)

        # Plotto i punti e salvo il grafico
        save_plot_with_progressive(sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals, 'r', 'r*', frase='Computed r* based on fixed r I order: r vs r*')

        # Salvo gli output dell'inversione
        salva_output_inversione_testuggine(r_vals, rstar_pullback_vals, 2)
        
        # Stampa riga bianca per tutta la larghezza della pagina
        stampa_riga_bianca()
        
        """
        #####################################################################
        Function inversion with distribution of r based on Taylor's truncation
        Second'ordine
        #####################################################################
        """
        print(f'II order inversion computation')
        # Priviamo ad utilizzare i primi due termini del polinomio di Taylor
        increment = lambdify(r, f / fdiff * (1 - sqrt(1 - 2 * fdiff * hstar)), 'numpy')
        r_vals = [rmin]
        rstar_vals = [rstarmin]
        rstar_pullback_vals = [rstarmin]
        hstarnum = int((rstarmax - rstarmin) / hstar)
        start_time = time.time()

        # Ricavo i valori di: r*, r ed r*_pullback
        rstar_vals, r_vals, rstar_pullback_vals = compila_valori_taylor(rstar_vals, r_vals, rstar_pullback_vals, hstarnum, increment)
        
        #print(f'rstar pullback num = {len(rstar_pullback_vals)}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Filtra i valori fino ad un massimo di max_points
        sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals = filtra_punti(r_vals, rstar_vals, rstar_pullback_vals)
 
        # Calcolo gli scarti tra raggi* adiacenti
        hstareff = calcola_differenze(rstar_pullback_vals)
        
        # Calcolo i momenti statistici degli r* pullback e li salva su un file .txt
        calcola_momenti_statistici_e_salva(hstar, hstareff, 3)

        # Plotto i punti e salvo il grafico
        save_plot_with_progressive(sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals, 'r', 'r*', frase='Computed r* based on fixed r II order: r vs r*')

        # Salva i risultati dell'inversione in un file di testo in una cartella
        salva_output_inversione_testuggine(r_vals, rstar_pullback_vals, 3)

        # Stampa riga bianca per tutta la larghezza della pagina
        stampa_riga_bianca()

        """
        #####################################################################
        Function inversion with distribution of r based on Taylor's truncation
        Terz'ordine
        #####################################################################
        """
        """
        print(f'III order inversion computation')
        # Priviamo ad utilizzare i primi tre termini del polinomio di Taylor
        f0 = f
        f1 = fdiff
        f2 = fddiff
        h = hstar
        sol = (sqrt((-2*(-f2 + 2*f1**2/f0**3)/f0
              + f1**2/f0**4)/(-f2 + 2*f1**2/f0**3)**2)*cos(atan(2*sqrt((9*h**2*(
              -f2 + 2*f1**2/f0**3)**2 + 2*(-f2 + 2*f1**2/f0**3)*(-9*f1*h/f0**2 + 4/f0**2)/f0
              - f1**2*(-6.0*f1*h/f0**2 + 3/f0**2)/f0**4)/(-f2 + 2*f1**2/f0**3)**4)*(-f2
              + 2*f1**2/f0**3)**3/(6*h*(-f2 + 2*f1**2/f0**3)**2 - 6*f1*(-f2
              + 2*f1**2/f0**3)/f0**3 + 2.0*f1**3/f0**6))/3))
        sol = simplify(expand(sol))
        increment = lambdify(r, sol, 'numpy')
        r_vals = [rmin]
        rstar_vals = [rstarmin]
        rstar_pullback_vals = [rstarmin]
        hstarnum = int((rstarmax - rstarmin) / hstar)
        start_time = time.time()

        # Ricavo i valori di: r*, r ed r*_pullback
        rstar_vals, r_vals, rstar_pullback_vals = compila_valori_taylor(rstar_vals, r_vals, rstar_pullback_vals, hstarnum, increment)
        
        #print(f'rstar pullback num = {len(rstar_pullback_vals)}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Filtra i valori fino ad un massimo di max_points
        sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals = filtra_punti(r_vals, rstar_vals, rstar_pullback_vals)
 
        # Calcolo gli scarti tra raggi* adiacenti
        hstareff = calcola_differenze(rstar_pullback_vals)
        
        # Calcolo i momenti statistici degli r* pullback e li salva su un file .txt
        calcola_momenti_statistici_e_salva(hstar, hstareff, 4)

        # Plotto i punti e salvo il grafico
        save_plot_with_progressive(sampled_r_vals, sampled_rstar_vals, sampled_rstar_pullback_vals, 'r', 'r*', frase='Computed r* based on fixed r III order: r vs r*')
        
        # Salva i risultati dell'inversione in un file di testo in una cartella
        salva_output_inversione_testuggine(r_vals, rstar_pullback_vals, 4)

        # Stampa riga bianca per tutta la larghezza della pagina
        stampa_riga_bianca()
        """
        
        
        
    except KeyboardInterrupt:
        # Gestisce l'interruzione del programma con Ctrl+C
        print("\nProgramma terminato.")


    
