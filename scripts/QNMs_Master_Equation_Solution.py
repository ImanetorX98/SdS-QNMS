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
import threading
import time
from inputimeout import inputimeout
import re as REX
import mpmath as mp
from rosignoli_lib import * 
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import traceback
import csv

mp.dps = 20

SHOW_PLOTS = os.getenv("QNMS_SHOW_PLOTS", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
if not SHOW_PLOTS:
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

_PLOT_COUNTERS = {}
_LAYOUT_WARNED = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)


def out_path(*parts):
    return os.path.join(OUT_DIR, *parts)


def normalize_output_folder(folder_name, create=False):
    if os.path.isabs(folder_name):
        resolved = folder_name
    else:
        normalized = folder_name.replace("\\", "/")
        if normalized == "out":
            resolved = OUT_DIR
        elif normalized.startswith("out/"):
            resolved = os.path.join(PROJECT_ROOT, normalized)
        else:
            resolved = os.path.join(OUT_DIR, folder_name)
    if create:
        os.makedirs(resolved, exist_ok=True)
    return resolved


def show_or_close():
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def safe_tight_layout():
    global _LAYOUT_WARNED
    try:
        plt.tight_layout()
    except Exception as exc:
        if not _LAYOUT_WARNED:
            print(f"Warning: tight_layout skipped due to non-finite plotting limits ({exc}).")
            _LAYOUT_WARNED = True


def to_real_float_array(values):
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    return np.asarray(arr, dtype=float)


def finite_xy(x_vals, y_vals, context="plot"):
    x_arr = to_real_float_array(x_vals).reshape(-1)
    y_arr = to_real_float_array(y_vals).reshape(-1)
    n = min(x_arr.shape[0], y_arr.shape[0])
    if n == 0:
        raise ValueError(f"No data available to plot {context}.")
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Not enough finite points to plot {context}.")
    return x_arr[mask], y_arr[mask]


def safe_relative_defect(reference, transformed, eps=1e-14):
    ref = np.abs(np.asarray(reference))
    diff = np.abs(np.asarray(reference) - np.asarray(transformed))
    out = np.zeros_like(diff, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(diff) & (ref > eps)
    out[mask] = diff[mask] / ref[mask]
    return out


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


def resolve_existing_path(name, require_dir=False):
    candidates = [
        out_path(name),
        os.path.join(PROJECT_ROOT, name),
        os.path.join(os.getcwd(), name),
        os.path.join(SCRIPT_DIR, name),
    ]
    for path in candidates:
        if require_dir and os.path.isdir(path):
            return path
        if not require_dir and os.path.isfile(path):
            return path
    kind = "directory" if require_dir else "file"
    raise ValueError(
        f"Required {kind} '{name}' not found. Checked: {candidates}. "
        "Run scripts/tortoise_inversion.py first to generate runtime inputs."
    )


# Funzione che stampa una riga bianca per aiutare la visualizzazione.
def stampa_riga_bianca():
    terminal_width, _ = shutil.get_terminal_size()
    # Stampa una riga piena con rich
    print("[white on white]" + "=" * terminal_width + "[/]")

"""
#####################################################################
Definizione della frequenza in funzione di n,l e della gravità superficiale
Con i metodi di approssimazione asintotici
#####################################################################
"""

# Frequenza per modi con alto l
def omega_high_l_nl(n,l):
    A = np.sqrt(1 - cosPar)
    B = 3 * np.sqrt(3) * M
    C = l + 1 / 2 - (n + 1 / 2) * 1j
    omega = A / B * C
    omega2 = omega ** 2
    return omega, omega2

# Frequenza per modi con alto n
def omega_high_n_nl(n,l):
    tempHawk = 1 / (8 * np.pi * M)
    #omega = ke / (2 * np.pi) * np.log(3) - 1j * ke * (n + 1 / 2)
    #omega = 1 / (4 * np.pi) * np.log(3) - 1j / 2 * (n - 1 / 2)
    omega = tempHawk * (- 2 * np.pi * 1j * (n + 1 / 2) + np.log(3))
    omega2 = omega ** 2
    return omega, omega2

"""
#####################################################################
Definizione della frequenza in funzione di n,l e delle derivate del potenziale
Con il metodo WKB al III ordine
#####################################################################
"""

# Since sign(Q^{(n>1)}) = - sing(V^{(n>1)}) only the odd combinations of Qs have been corrected
def Lambda_norm(alpha, Vns): # Sistema il codice che qualcosa è errato
    V0, V2, V3, V4, V5, V6 = Vns
    A = 1 / np.sqrt(- 2 * V2)
    B = (1 / 8 * (V4 / V2) * (1 / 4 + alpha ** 2)
         - 1 / 288 * (V3 / V2) ** 2 * (7 + 60 * alpha ** 2))
    return A * B

def Omega_norm(alpha, Vns): # Sistema il codice che qualcosa è errato
    V0, V2, V3, V4, V5, V6 = Vns
    A = alpha / (- 2 * V2)
    B = (5 / 6912 * (V3 / V2) ** 4 * (77 + 188 * alpha ** 2)
         - 1 / 384 * ( (- V3) ** 2 * (- V4) / (- V2) ** 3) * (51 + 100 * alpha ** 2))
    C = (1 / 2304 * (V4 / V2) ** 2 * (67 * alpha ** 2)
         + 1 / 288 * (V3 * V5 / V2 ** 2) * (19 + 28 * alpha ** 2)
         - 1 / 288 * (V6 / V0) * (5 + 4 * alpha ** 2))
    return A * (B + C)
    
def frequency(n,l):
    global r0
    alpha = n + 1 / 2
    r0, defect = maximum_potential(V1eff, V2eff, V3eff, l, re, rc, M)
    V0 = V0eff(r0,l)
    # Si pone ad hoc per scegliere D(V) rispetto r o r*
    starPotential = True 
    if starPotential:
        V2 = V2effStar(r0,l)
        V3 = V3effStar(r0,l)
        V4 = V4effStar(r0,l)
        V5 = V5effStar(r0,l)
        V6 = V6effStar(r0,l)
    else:
        V2 = V2eff(r0,l)
        V3 = V3eff(r0,l)
        V4 = V4eff(r0,l)
        V5 = V5eff(r0,l)
        V6 = V6eff(r0,l)
        
    Vns = [V0, V2, V3, V4, V5, V6]
    Lambda_ = Lambda_norm(alpha, Vns)
    Omega_ = Omega_norm(alpha, Vns)
    # + Omega_
    omega2 = V0 - 1j * (alpha  + Lambda_ + Omega_) * np.sqrt(- 2 * V2)
    omega = np.sqrt(omega2)
    return omega, omega2

"""
#####################################################################
Funzioni di ausilio alla computazione
Chiedi input a tempo, carica dati, ignora avvisi e calcolo differenze.
#####################################################################
"""

def chiedi_input_time(frase = 'Set the value before timeout : ', value = 10):
    # Si prova a settare n
    try:
    # Take timed input using inputimeout() function
        val = int(inputimeout(prompt = frase, timeout=10))
    # Catch the timeout error
    except Exception:
        print(f"Timeout. Set value on {value}")
    # Declare the timeout statement
        val = value
    return val
        
def load_data(file_name):
    try:
        data = np.loadtxt(file_name, delimiter=' ')
        x = data[:, 0]  # r
        y = data[:, 1]  # rStar
        #length = x.shape[0]
        print(f"Data from {file_name} imported successfully.")
        return x, y
    except Exception as e:
        raise ValueError(f"Unable to load runtime data file '{file_name}': {e}") from e

def ignore_warnings():
    import warnings
    warnings.filterwarnings("ignore")


def compute_wkb_from_derivatives(n_mode, V0, V2, V3, V4, V5, V6):
    alpha = n_mode + 0.5
    Vns = [V0, V2, V3, V4, V5, V6]
    lambda_ = Lambda_norm(alpha, Vns)
    omega_ = Omega_norm(alpha, Vns)
    omega2 = V0 - 1j * (alpha + lambda_ + omega_) * np.sqrt(-2 * V2)
    omega = np.sqrt(omega2)
    return complex(omega), complex(omega2)


def estimate_wkb_from_interpolation_windows(n_mode, l_mode, r_vals, rstar_vals, margins=(3, 4, 5, 6)):
    omega_samples = []
    r_arr = np.asarray(r_vals, dtype=float)
    rstar_arr = np.asarray(rstar_vals, dtype=float)

    for margin in margins:
        try:
            V0, V2, V3, V4, V5, V6 = interpolatePotential(
                V0eff, l_mode, V1eff, V2eff, V3eff, re, rc, M, r_arr, rstar_arr, margin=margin
            )
            omega, _ = compute_wkb_from_derivatives(n_mode, V0, V2, V3, V4, V5, V6)
            if np.isfinite(np.real(omega)) and np.isfinite(np.imag(omega)):
                omega_samples.append(omega)
        except Exception:
            continue

    if not omega_samples:
        omega_fallback, _ = frequency(n_mode, l_mode)
        return complex(omega_fallback), 0.0, 0.0, 1

    samples = np.asarray(omega_samples, dtype=complex)
    omega_mean = complex(np.mean(samples))
    err_re = float(np.std(np.real(samples))) if samples.size > 1 else 0.0
    err_im = float(np.std(np.imag(samples))) if samples.size > 1 else 0.0
    return omega_mean, err_re, err_im, int(samples.size)


def downsample_grid(r_vals, rstar_vals, max_points=1800):
    r_arr = np.asarray(r_vals, dtype=float).reshape(-1)
    rstar_arr = np.asarray(rstar_vals, dtype=float).reshape(-1)
    length = min(r_arr.size, rstar_arr.size)
    r_arr = r_arr[:length]
    rstar_arr = rstar_arr[:length]
    if length <= max_points:
        return r_arr, rstar_arr
    idx = np.linspace(0, length - 1, int(max_points), dtype=int)
    idx = np.unique(idx)
    return r_arr[idx], rstar_arr[idx]


def collect_tortoise_variants(output_folder, m_target, l_target, inversion_order=3):
    pattern = REX.compile(
        rf"^output_{inversion_order}_([^_]+)_([^_]+)_([^_]+)_([^_]+)_\.txt$"
    )
    variants = []
    for filename in os.listdir(output_folder):
        match = pattern.match(filename)
        if not match:
            continue
        try:
            m_val = float(match.group(1))
            l_val = float(match.group(2))
            hrad_val = float(match.group(3))
            hstar_val = float(match.group(4))
        except ValueError:
            continue
        if not np.isclose(m_val, m_target, rtol=1e-10, atol=1e-12):
            continue
        if not np.isclose(l_val, l_target, rtol=1e-10, atol=1e-12):
            continue
        variants.append(
            {
                "file": filename,
                "path": os.path.join(output_folder, filename),
                "M": m_val,
                "L": l_val,
                "hrad": hrad_val,
                "hstar": hstar_val,
            }
        )
    variants.sort(key=lambda item: (item["hstar"], item["hrad"]))
    return variants


def evenly_spaced_subset(items, max_items):
    if len(items) <= max_items:
        return list(items)
    idx = np.linspace(0, len(items) - 1, int(max_items), dtype=int)
    idx = np.unique(idx)
    return [items[i] for i in idx]


def compute_time_domain_mode(r_vals, rstar_vals, l_mode, space_time_k, p_prony=6, max_points=1800):
    r_sub, rstar_sub = downsample_grid(r_vals, rstar_vals, max_points=max_points)
    i0, phi, t, dim, rstar_uv, r_uv = pde_solution(
        r_sub, rstar_sub, l_mode, space_time_k, max_pde_points=max_points
    )
    profile, t_plot = sliceTimeProfile(0, phi, t, dim)
    used_fallback = False
    try:
        omega, err_re, err_im, windows_used = estimate_mode_with_windows(
            profile, t_plot, p=p_prony, verbose=False
        )
    except Exception as exc:
        omega = estimate_mode_fft_fallback(profile, t_plot)
        err_re = float("nan")
        err_im = float("nan")
        windows_used = 0
        used_fallback = True
        print(f"Warning: Prony window estimation failed ({exc}). Falling back to FFT envelope estimate.")
    t_plot = np.asarray(t_plot, dtype=float)
    dt_vals = np.diff(t_plot)
    finite_dt = dt_vals[np.isfinite(dt_vals) & (dt_vals != 0)]
    dt = float(np.median(np.abs(finite_dt))) if finite_dt.size > 0 else float("nan")
    return {
        "omega": complex(omega),
        "err_re": float(err_re),
        "err_im": float(err_im),
        "windows": int(windows_used),
        "used_fallback": bool(used_fallback),
        "points_used": int(r_sub.size),
        "dt": dt,
    }


def save_csv_rows(file_path, fieldnames, rows):
    with open(file_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_method_cross_check(n_mode, l_mode, r_vals, rstar_vals, include_time_domain=False,
                           space_time_k=5, p_prony=6, max_points=1800,
                           save_folder="QNMS_METHOD_CROSSCHECK"):
    save_folder = normalize_output_folder(save_folder, create=True)
    omega_wkb, _ = frequency(n_mode, l_mode)
    omega_frob, _ = omega_frobenius(re, rc, ro, M, l_mode, n_mode, s)
    omega_wkb = complex(omega_wkb)
    omega_frob = complex(omega_frob)

    eps = 1e-15
    rows = [
        {
            "method": "WKB_III",
            "n": n_mode,
            "l": l_mode,
            "spaceTimeK": "",
            "omega_re": float(np.real(omega_wkb)),
            "omega_im": float(np.imag(omega_wkb)),
            "err_re": 0.0,
            "err_im": 0.0,
            "relative_to_WKB": 0.0,
        },
        {
            "method": "Frobenius_Leaver_like",
            "n": n_mode,
            "l": l_mode,
            "spaceTimeK": "",
            "omega_re": float(np.real(omega_frob)),
            "omega_im": float(np.imag(omega_frob)),
            "err_re": 0.0,
            "err_im": 0.0,
            "relative_to_WKB": float(abs(omega_frob - omega_wkb) / max(abs(omega_wkb), eps)),
        },
    ]

    print(f'Cross-check WKB vs Frobenius:')
    print(f'  WKB( n={n_mode}, l={l_mode} ) = {omega_wkb}')
    print(f'  Frobenius( n={n_mode}, l={l_mode} ) = {omega_frob}')
    print(f'  Relative gap = {rows[1]["relative_to_WKB"]}')

    if include_time_domain:
        try:
            td = compute_time_domain_mode(
                r_vals, rstar_vals, l_mode, space_time_k, p_prony=p_prony, max_points=max_points
            )
            omega_td = td["omega"]
            rel_td = float(abs(omega_td - omega_wkb) / max(abs(omega_wkb), eps))
            method_name = "TimeDomain_FFTFallback" if td.get("used_fallback", False) else "TimeDomain_Prony"
            rows.append(
                {
                    "method": method_name,
                    "n": n_mode,
                    "l": l_mode,
                    "spaceTimeK": int(space_time_k),
                    "omega_re": float(np.real(omega_td)),
                    "omega_im": float(np.imag(omega_td)),
                    "err_re": td["err_re"],
                    "err_im": td["err_im"],
                    "relative_to_WKB": rel_td,
                }
            )
            print(f'  Time-domain( K={space_time_k} ) = {omega_td} ± ({td["err_re"]}, {td["err_im"]})')
            print(f'  Relative gap TD/WKB = {rel_td}')
        except Exception as exc:
            print(f'  Time-domain estimate failed and was skipped: {exc}')
            rows.append(
                {
                    "method": "TimeDomain_Failed",
                    "n": n_mode,
                    "l": l_mode,
                    "spaceTimeK": int(space_time_k),
                    "omega_re": float("nan"),
                    "omega_im": float("nan"),
                    "err_re": float("nan"),
                    "err_im": float("nan"),
                    "relative_to_WKB": float("nan"),
                }
            )

    output_file = os.path.join(
        save_folder, f"crosscheck_{M}_{L}_n{n_mode}_l{l_mode}_{int(time.time())}.csv"
    )
    save_csv_rows(
        output_file,
        ["method", "n", "l", "spaceTimeK", "omega_re", "omega_im", "err_re", "err_im", "relative_to_WKB"],
        rows,
    )
    print(f'Cross-check report saved in {output_file}')
    return output_file


def run_convergence_error_analysis(n_mode, l_mode, space_time_ks, p_prony=6, max_variants=6,
                                   max_points=1800, save_folder="QNMS_CONVERGENCE_ANALYSIS",
                                   base_r=None, base_rstar=None):
    save_folder = normalize_output_folder(save_folder, create=True)
    output_folder = resolve_existing_path("TORTOISE_INVERSION_OUTPUT", require_dir=True)
    variants = collect_tortoise_variants(output_folder, M, L, inversion_order=3)
    variants = evenly_spaced_subset(variants, max_variants)
    if len(variants) == 0:
        if base_r is None or base_rstar is None:
            raise ValueError(
                f"No output_3 inversion files found for M={M}, L={L}, and no fallback grid was provided."
            )
        print(
            "Warning: no output_3 inversion files found. "
            "WKB resolution sweep will use current in-memory grid only."
        )
    elif len(variants) == 1:
        print(
            "Warning: only 1 output_3 inversion file found. "
            "Convergence order in hstar/hrad cannot be estimated."
        )
    else:
        print(f"Convergence analysis using {len(variants)} inversion variants.")

    wkb_rows = []
    if variants:
        for var in variants:
            x_var, y_var = load_data(var["path"])
            omega_est, err_re, err_im, sample_count = estimate_wkb_from_interpolation_windows(
                n_mode, l_mode, x_var, y_var
            )
            wkb_rows.append(
                {
                    "file": var["file"],
                    "hrad": var["hrad"],
                    "hstar": var["hstar"],
                    "omega_re": float(np.real(omega_est)),
                    "omega_im": float(np.imag(omega_est)),
                    "err_re": err_re,
                    "err_im": err_im,
                    "interp_samples": int(sample_count),
                    "abs_error": float("nan"),
                    "rel_error": float("nan"),
                    "order_to_finer": float("nan"),
                }
            )
    else:
        omega_est, err_re, err_im, sample_count = estimate_wkb_from_interpolation_windows(
            n_mode, l_mode, base_r, base_rstar
        )
        wkb_rows.append(
            {
                "file": "in_memory_grid_x3_y3",
                "hrad": float(hstar),
                "hstar": float(hstar),
                "omega_re": float(np.real(omega_est)),
                "omega_im": float(np.imag(omega_est)),
                "err_re": err_re,
                "err_im": err_im,
                "interp_samples": int(sample_count),
                "abs_error": 0.0,
                "rel_error": 0.0,
                "order_to_finer": float("nan"),
            }
        )

    ref_row = min(wkb_rows, key=lambda row: (row["hstar"], row["hrad"])) if wkb_rows else None
    omega_ref = complex(ref_row["omega_re"], ref_row["omega_im"]) if ref_row else complex(np.nan, np.nan)
    eps = 1e-15
    if wkb_rows and np.isfinite(np.real(omega_ref)) and np.isfinite(np.imag(omega_ref)):
        for row in wkb_rows:
            omega_curr = complex(row["omega_re"], row["omega_im"])
            abs_err = abs(omega_curr - omega_ref)
            row["abs_error"] = float(abs_err)
            row["rel_error"] = float(abs_err / max(abs(omega_ref), eps))

    rows_sorted = sorted(wkb_rows, key=lambda row: row["hstar"], reverse=True)
    if len(rows_sorted) >= 2:
        for i in range(len(rows_sorted) - 1):
            h1 = rows_sorted[i]["hstar"]
            h2 = rows_sorted[i + 1]["hstar"]
            e1 = rows_sorted[i]["abs_error"]
            e2 = rows_sorted[i + 1]["abs_error"]
            if h1 > h2 and e1 > 0 and e2 > 0:
                rows_sorted[i]["order_to_finer"] = float(np.log(e1 / e2) / np.log(h1 / h2))

    wkb_csv = os.path.join(
        save_folder, f"wkb_resolution_{M}_{L}_n{n_mode}_l{l_mode}_{int(time.time())}.csv"
    )
    save_csv_rows(
        wkb_csv,
        ["file", "hrad", "hstar", "omega_re", "omega_im", "err_re", "err_im",
         "interp_samples", "abs_error", "rel_error", "order_to_finer"],
        rows_sorted,
    )
    print(f'WKB convergence report saved in {wkb_csv}')

    # spaceTimeK sweep on the finest available inversion grid
    if variants:
        finest_variant = min(variants, key=lambda item: (item["hstar"], item["hrad"]))
        x_ref, y_ref = load_data(finest_variant["path"])
    else:
        x_ref = np.asarray(base_r, dtype=float)
        y_ref = np.asarray(base_rstar, dtype=float)

    td_rows = []
    for k_value in sorted(set(int(k) for k in space_time_ks)):
        try:
            td = compute_time_domain_mode(
                x_ref, y_ref, l_mode, k_value, p_prony=p_prony, max_points=max_points
            )
            td_rows.append(
                {
                    "spaceTimeK": int(k_value),
                    "dt": td["dt"],
                    "points_used": td["points_used"],
                    "windows_used": td["windows"],
                    "omega_re": float(np.real(td["omega"])),
                    "omega_im": float(np.imag(td["omega"])),
                    "err_re": td["err_re"],
                    "err_im": td["err_im"],
                    "abs_error": float("nan"),
                    "rel_error": float("nan"),
                    "order_to_finer": float("nan"),
                }
            )
        except Exception as exc:
            print(f"Warning: time-domain sweep failed for spaceTimeK={k_value}: {exc}")
            td_rows.append(
                {
                    "spaceTimeK": int(k_value),
                    "dt": float("nan"),
                    "points_used": 0,
                    "windows_used": 0,
                    "omega_re": float("nan"),
                    "omega_im": float("nan"),
                    "err_re": float("nan"),
                    "err_im": float("nan"),
                    "abs_error": float("nan"),
                    "rel_error": float("nan"),
                    "order_to_finer": float("nan"),
                }
            )

    valid_rows = [
        row for row in td_rows
        if np.isfinite(row["dt"]) and np.isfinite(row["omega_re"]) and np.isfinite(row["omega_im"])
    ]
    if valid_rows:
        td_ref = min(valid_rows, key=lambda row: row["dt"])
        omega_td_ref = complex(td_ref["omega_re"], td_ref["omega_im"])
        for row in valid_rows:
            omega_curr = complex(row["omega_re"], row["omega_im"])
            abs_err = abs(omega_curr - omega_td_ref)
            row["abs_error"] = float(abs_err)
            row["rel_error"] = float(abs_err / max(abs(omega_td_ref), eps))

        valid_sorted = sorted(valid_rows, key=lambda row: row["dt"], reverse=True)
        for i in range(len(valid_sorted) - 1):
            dt1 = valid_sorted[i]["dt"]
            dt2 = valid_sorted[i + 1]["dt"]
            e1 = valid_sorted[i]["abs_error"]
            e2 = valid_sorted[i + 1]["abs_error"]
            if dt1 > dt2 and e1 > 0 and e2 > 0:
                valid_sorted[i]["order_to_finer"] = float(np.log(e1 / e2) / np.log(dt1 / dt2))
    else:
        print("Warning: no valid time-domain samples were extracted for the selected spaceTimeK sweep.")

    td_sorted = sorted(
        td_rows,
        key=lambda row: (0 if np.isfinite(row["dt"]) else 1, -row["dt"] if np.isfinite(row["dt"]) else 0)
    )

    td_csv = os.path.join(
        save_folder, f"time_domain_k_sweep_{M}_{L}_n{n_mode}_l{l_mode}_{int(time.time())}.csv"
    )
    save_csv_rows(
        td_csv,
        ["spaceTimeK", "dt", "points_used", "windows_used", "omega_re", "omega_im",
         "err_re", "err_im", "abs_error", "rel_error", "order_to_finer"],
        td_sorted,
    )
    print(f'Time-domain convergence report saved in {td_csv}')
    return wkb_csv, td_csv

def psi_asinth(omega2, horizon=False, l=1, rStar = -1000):
    Q0 = (omega2-V0eff(rStar,l))
    r = symbols('r')
    Q0x = lambdify((r, l), omega2 - V, 'numpy')
    S1 = - 1 / 4 * np.log(Q0)
    S3 = 1 / 16 * ( - V2eff(rStar,l) / Q0 ** 2 - 5 / 4 * V1eff(rStar,l) ** 2 / Q0 ** 3)
    if horizon:
        S0 = - 1j * integrale(np.sqrt(Q0x(x,l)))
        S2 = + 1j * integrale( - V2eff(x,l) / Q0x(x,l) ** (3/2) - 5 / 4 * V1eff(x,l) ** 2 / Q0x(x,l) ** (5/2))
    else:
        S0 = + 1j * integrale(np.sqrt(Q0x(x,l)))
        S2 = - 1j * integrale( - V2eff(x,l) / Q0x(x,l) ** (3/2) - 5 / 4 * V1eff(x,l) ** 2 / Q0x(x,l) ** (5/2))
    psi = np.exp(S0 + S1 + S2 + S3)
    return psi

"""
#####################################################################
Implementazione dell'algoritmo di Gundlach-Price-Pullin
Per l'equazione [4 * d^2 / dudv + V] * R(u,v) = 0
#####################################################################
"""

# diamond_borders_construction è in qnms_utils
def pde_solution(r,rStar,l,spaceTimeK,max_pde_points=None):
    r = np.asarray(r, dtype=float).reshape(-1)
    rStar = np.asarray(rStar, dtype=float).reshape(-1)
    n = min(r.size, rStar.size)
    r = r[:n]
    rStar = rStar[:n]

    # Keep PDE grid tractable on very dense inversion outputs.
    if max_pde_points is None:
        max_pde_points = int(os.getenv("QNMS_PDE_MAX_POINTS", "6000"))
    else:
        max_pde_points = int(max(200, max_pde_points))
    if n > max_pde_points:
        idx = np.linspace(0, n - 1, max_pde_points, dtype=int)
        idx = np.unique(idx)
        r = r[idx]
        rStar = rStar[idx]
        print(f"PDE grid downsampled from {n} to {len(idx)} points (PDE_MAX_POINTS={max_pde_points}).")

    phiU, phiV, u, v, t, rStarUV, rUV, hStarUV, i0, hStarM = diamond_borders_construction(r,rStar,spaceTimeK)
    dim = int((len(rStarUV)+1)/2)
    if dim < 2:
        raise ValueError(
            f"Invalid PDE grid after symmetric r* projection: dim={dim}, "
            f"len(rStarUV)={len(rStarUV)}. Provide a denser/wider tortoise grid."
        )
    phi = np.zeros((dim,dim))
    phi[:,0] = phiV.T 
    phi[0,:] = phiU

    # Precompute potential terms on the (u,v) grid to avoid repeated calls.
    idx_i = np.arange(1, dim)
    idx_j = np.arange(1, dim)
    idx_grid = i0 + idx_i[:, None] - idx_j[None, :]
    potential_grid = 1 - 0.125 * (hStarM * spaceTimeK) ** 2 * V0eff(r[idx_grid], l)

    for i in tqdm(range(1, dim), desc="PDE Progress", leave=False):
        for j in range(1, dim):
            potentialTerm = potential_grid[i - 1, j - 1]
            phi[i, j] = (phi[i, j - 1] + phi[i - 1, j]) * potentialTerm - phi[i - 1, j - 1]
    return i0, phi, t, dim, rStarUV, rUV
    
def timeProfile(indexDiff, phi, t, dim):
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

def spaceProfile(indexTime, phi, rStar, dim):
    # I extract the space profile
    # I need to gather all the elements
    # on a diagonal and put them into a numpy array
    # indexTime è un numero compreso tra 0 e C, divido i casi in base
    # al segno di indexTime - i0.
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
    if indexMod == 0:
        rStarPlot = np.array(rStar)[::2]
    else:
        rStarPlot = rStar[indexMod:-indexMod]
        rStarPlot = rStarPlot[::2]
    return spaceProfile, rStarPlot
    
"""
#####################################################################
Implementazione dell'algoritmo di RKIV
Per l'equazione [d^2/dr*^2 + (omega^2 - V)] * R = 0, d/dr* R = T
#####################################################################
"""

# Value of dR/dr*
def g1(R, Rder, rStar, r, l, omega2):
    return Rder

# Value of d^2R/dr*^2
def g2(R, Rder, rStar, r, l, omega2, even=False):
    dT = (V0effE(r, l) - omega2) * R if even else (V0eff(r, l) - omega2) * R
    return dT # Il potenziale è funzione di r

def rk4Special(r,rStar,l,n,highLOrNot=False,
        highNOrNot=False,backWard=False,even=False,frobCorr=False):
    I = r.shape[0]
    ym = np.zeros((2, I), dtype=complex)

    # Frequency definition in base of the method
    omega, omega2 = omega_high_l_nl(n,l) if highLOrNot else (omega_high_n_nl(n,l) if highNOrNot else frequency(n,l))
    indexStar = indice_valore_piu_vicino(x3, r0)
    r0Star = y3[indexStar]
    ufunc, u = funzione_u(re,rc,ro,M,l,omega,s)
    x = symbols('x')
    uder = diff(u, x)
    uderl = lambdify(x,uder,'numpy')
    
    # Seleziono il metodo (onWard/backWard) per il calcolo della funzione
    r = r[::-1] if backWard else r
    rStar = rStar[::-1] if backWard else rStar
    potential = V0effE(r, l) if even else V0eff(r, l)
    uBegin = ufunc(np.reciprocal(r[0]))
    uDerBegin = uderl(np.reciprocal(r[0]))
    if frobCorr:
        ym[0,0] = np.exp(1j * omega * rStar[0]) if backWard else np.exp(-1j * omega * rStar[0]) * uBegin
        #ym[1,0] = (1j * omega * ym[0,0]) if backWard else - 1j * omega * uBegin * ym[0,0]
        ym[1,0] = (1j * omega * ym[0,0]) if backWard else ym[0,0] * (- 1j * omega * uBegin - uDerBegin / r[0] ** 2 * fl (r[0]))
    else:
        ym[0,0] = np.exp(1j * omega * rStar[0]) if backWard else np.exp(-1j * omega * rStar[0])
        ym[1,0] = (1j * omega * ym[0,0]) if backWard else - 1j * omega * ym[0,0]
    
    # Implemento il RKII per individuare il primo step
    h = rStar[1] - rStar[0]

    R = ym[0,0]
    T = ym[1,0]
    
    k11 = h * T
    k21 = h * (potential[0] - omega2) * R

    k12 = h * T
    k22 = h * (potential[1] - omega2) * R
    
    ym[0,1] = ym[0,0] + (1 / 2) * (k11 + k12)
    ym[1,1] = ym[1,0] + (1 / 2) * (k12 + k22)
    first_step_norm = max(abs(ym[0, 1]), abs(ym[1, 1]))
    if np.isfinite(first_step_norm) and first_step_norm > 1e150:
        ym[:, 1] /= first_step_norm
    elif not np.isfinite(first_step_norm):
        ym[:, 1] = ym[:, 0]
    
    # Si implementa l'algoritmo di Runge Kutta al IV ordine
    for i in tqdm(range(I-2), desc="RK4 Progress", leave=False):
        # Incremento in r*
        
        h = rStar[i+2] - rStar[i]
            
        R = ym[0,i] # FUNZIONE
        T = ym[1,i] # DERIVATA STAR
            
        k11 = h * T
        k21 = h * (potential[i] - omega2) * R
            
        R12 = R + k11 / 2
        T12 = T + k21 / 2
        k12 = h * T12
        k22 = h * (potential[i + 1] - omega2) * R12
            
        R13 = R + k12 / 2
        T13 = T + k22 / 2
        k13 = h * T13
        k23 = h * (potential[i + 1] - omega2) * R13
            
        R14 = R + k13
        T14 = T + k23
        k14 = h * T14
        k24 = h * (potential[i + 2] - omega2) * R14
            
        if i < I-1:
            ym[0, i+1] = R + 1 / 6 * (k11 + 2 * (k12 + k13) + k14) # R
            ym[1, i+1] = T + 1 / 6 * (k21 + 2 * (k22 + k23) + k24) # T
            step_norm = max(abs(ym[0, i + 1]), abs(ym[1, i + 1]))
            if np.isfinite(step_norm) and step_norm > 1e150:
                ym[:, i + 1] /= step_norm
            elif not np.isfinite(step_norm):
                ym[:, i + 1] = ym[:, i]
            
    ym[0,I-1] = ym[0,I-2]
    ym[1,I-1] = ym[1,I-2]
                
    # Trasponi la matrice per scrivere le righe in verticale
    #ym = ym.T
    if backWard:
        ym = ym.T
        ym = ym[::-1]
        ym = ym.T
        
    return ym

def plot(x,ym, frase = 'QNMs with root finding and interpolation'):
    y = ym[0, :] / x[::2]
    # Creazione del grafico
    plt.plot(x[::2], y, label='y = QNM')      # Plot della curva
    plt.xlabel('x')                      # Etichetta asse x
    plt.ylabel('y')                      # Etichetta asse y
    plt.title(frase)                     # Titolo del grafico
    plt.legend()                         # Mostra la legenda
    plt.grid(True)                       # Mostra la griglia
    show_or_close()

def estremali_k_percento(lista, perc):
    percentuale = 10**(-2) * perc
    num_elementi_da_stampare = int(len(lista) * percentuale)
    ultimi_elementi = lista[-num_elementi_da_stampare:]
    num_elementi_da_stampare = int(len(lista) * percentuale)
    primi_elementi = lista[:num_elementi_da_stampare]
    return primi_elementi, ultimi_elementi

# Transforms ys for semilogy plot
def y_semilogy(array):
    absY = np.abs(np.asarray(array, dtype=complex))
    finite_mask = np.isfinite(absY)
    if not np.any(finite_mask):
        raise ValueError("Semilogy transform failed: all values are NaN/Inf.")
    scale = np.max(absY[finite_mask])
    if scale <= 0:
        out = np.full(absY.shape, np.finfo(float).tiny, dtype=float)
        out[~finite_mask] = np.nan
        return out
    out = np.full(absY.shape, np.nan, dtype=float)
    out[finite_mask] = absY[finite_mask] / scale * 10
    out[(out <= 0) & finite_mask] = np.finfo(float).tiny
    return out

def save_plot_with_progressive(x, xStar, ym, labelx='x', labely='y', frase='QNMs', plotTortoise=False, l=1,
                               plotPotential=False, n=0, monoMode=False, semiLog=False,
                               save_folder='QNMS_PLOTS', special=False):
    # Modificare il plot per poter plottare il potenziale in termini di r e non r*, questo richiede il caricamento di entrame le liste corrispondenti.
    # Una volta eseguito il caricamento si plotta il grafo di V in termini di r vs r oppure r*.
    xNormPlot = to_real_float_array(x if special else x[::-2])
    xStarPlot = to_real_float_array(xStar if special else xStar[::-2])
    yRe = np.real(np.asarray(ym[0, :], dtype=complex))
    yIm = np.imag(np.asarray(ym[0, :], dtype=complex))
    min_len = min(len(xNormPlot), len(xStarPlot), len(yRe), len(yIm))
    if min_len < 2:
        raise ValueError(f"Not enough points to plot '{frase}'.")
    xNormPlot = xNormPlot[:min_len]
    xStarPlot = xStarPlot[:min_len]
    yRe = yRe[:min_len]
    yIm = yIm[:min_len]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        fl_vals = to_real_float_array(fl(xNormPlot))
        safe_ratio = np.divide(
            xNormPlot,
            fl_vals,
            out=np.zeros_like(xNormPlot, dtype=float),
            where=np.isfinite(fl_vals) & (np.abs(fl_vals) > 1e-14)
        )
        yRe = yRe * safe_ratio
    finite_y = yRe[np.isfinite(yRe)]
    if finite_y.size == 0:
        raise ValueError(f"All computed values are non-finite for plot '{frase}'.")
    max_abs = np.max(np.abs(finite_y))
    if max_abs > 1e6:
        yRe = yRe / max_abs
    yMax = np.max(np.abs(finite_y))
    xPlot = xStarPlot if plotTortoise else xNormPlot
    
    # I initialize the zero point value for semilogy plot
    parZero = 0 #.000001
    
    # Creazione del grafico
    #fig, ax = plt.subplots(figsize=(10, 7))
    if monoMode: # Le onde in e out si plottano in funzione di xStar
        omega, omega2 = frequency(n,l)
        inNormXs, outNormXs = estremali_k_percento(x,10)
        inStarXs, outStarXs = estremali_k_percento(xStar,10)
        inYs, outYs = estremali_k_percento(np.real(ym[0,:]),10)
        print(f'maxYin = {max(inYs)}, maxYout = {max(outYs)}')
        outWaveYs = np.real(np.exp(1j * omega * outStarXs))
        inWaveYs = np.real(np.exp(-1j * omega * inStarXs))
        
        # Near horizon waves correction
        #ufunc, u = funzione_u(re,rc,ro,M,lExp,omega,s)
        #uYs = ufunc(inNormXs)
        
        plotInXs = inStarXs if plotTortoise else inNormXs
        plotOutXs = outStarXs if plotTortoise else outNormXs
        #inWaveYs = (abs(inWaveYs * uYs) + parZero) * 10 / (max(inWaveYs * uYs) + parZero) if semiLog else inWaveYs

        #inWaveYs = (abs(inWaveYs) + parZero) * 10 / (max(inWaveYs) + parZero) if semiLog else inWaveYs
        #outWaveYs = (abs(outWaveYs) + parZero) * max(outYs) / yMax / (max(outWaveYs) + parZero) if semiLog else outWaveYs
        inXVals, inYVals = finite_xy(plotInXs, y_semilogy(inWaveYs) if semiLog else inWaveYs, f"{frase} inWave")
        outXVals, outYVals = finite_xy(plotOutXs, y_semilogy(outWaveYs) if semiLog else outWaveYs, f"{frase} outWave")
        plt.semilogy(inXVals, inYVals, label='y = inWave') if semiLog else plt.plot(inXVals, inYVals, label='y = inWave')
        plt.semilogy(outXVals, outYVals, label='y = outWave') if semiLog else plt.plot(outXVals, outYVals, label='y = outWave')

    if monoMode:
        omega, omega2 = frequency(n,l)
        indexStar = indice_valore_piu_vicino(x, r0)
        r0Star = xStar[indexStar]
        A = 1
        B = 1
        
        #psiInYs, psiOutYs = psi_In_Out(V,f,r0,r0Star,ni,l,omega,omega2,A,B,x,xStar)
        
    if plotPotential: # Il potenziale si plotta in coordinate r
        potentialYs = V0eff(xNormPlot, l)
        potentialYs = to_real_float_array(potentialYs)
        finite_p = potentialYs[np.isfinite(potentialYs)]
        if finite_p.size > 0:
            potentialYsMax = np.max(np.abs(finite_p))
            if potentialYsMax > 0 and not semiLog:
                potentialYs = potentialYs * (yMax / potentialYsMax if yMax > 0 else 1.0)
            xPot, yPot = finite_xy(xPlot, y_semilogy(potentialYs) if semiLog else potentialYs, f"{frase} potential")
            plt.semilogy(xPot, yPot, label='y = Potential') if semiLog else plt.plot(xPot, yPot, label='y = Potential')

    #yPlot = (abs(yRe) + parZero) * 10 / (yMax + parZero) if semiLog else yRe
    xQNM, yQNM = finite_xy(xPlot, y_semilogy(yRe) if semiLog else yRe, f"{frase} Re(QNM)")
    plt.semilogy(xQNM, yQNM, label='y = Re QNM') if semiLog else plt.plot(xQNM, yQNM, label='y = Re QNM')

    labelx = 'r*' if plotTortoise else 'r'
    
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(frase)
    plt.legend()
    safe_tight_layout()
    plt.grid(True)

    save_folder = normalize_output_folder(save_folder, create=True)

    progressive_pattern = REX.compile(r'^QNM_(\d+)\.png$')
    filename = next_progressive_plot_path(save_folder, progressive_pattern, "QNM_")

    # Salvataggio del plot
    plt.savefig(filename)
    print(f'Plot saved in {filename}')

    # Mostra il plot
    show_or_close()

if __name__ == "__main__":
    try:
        global M
        global L
        global hstar
        global re
        global rc
        global ro
        global ke
        global cosPar
        global V
        global Ve
        
        # Symbols' definition
        w, t, r, V, Ve, M, A, B, C, L, c, dK, beta = symbols('w t r V Ve M A B C L c dK beta', real = True)
        l, s = symbols('l s', integer = True)
        re, rc, ro, rstar = symbols('re rc ro rstar', real = True)
        ke, kc, ko = symbols('ke kc ko', real = True)
        
        # Leggo il file dei parametri per poter poi caricare i dati
        params_path = resolve_existing_path("qnms_parameters_.txt")
        lines = rl.read_file(params_path)

        # White line print
        stampa_riga_bianca()

        # Inizializzo i valori delle costanti del programma
        M = float(lines[0])
        print(f'M = {M}')
        L = float(lines[1])
        cosPar = 9 * M ** 2 * L
        Lambda = unicodedata.lookup('GREEK CAPITAL LETTER LAMDA')
        PsiGreek = unicodedata.lookup('GREEK CAPITAL LETTER PSI')
        omegaGreek = unicodedata.lookup('GREEK SMALL LETTER OMEGA')
        print(f'{Lambda} = {L}')
        hrad = float(lines[2])
        print(f'hrad = {hrad}')
        hstar = float(lines[3])
        print(f'hstar = {hstar}')
        letter = (lines[4]).strip()

        rootFinding = True if letter == 'S' else False
        print(f'rootFinding = {rootFinding}')
       
        re = float(lines[5])
        print(f're = {re}')
        rc = float(lines[6])
        print(f'rc = {rc}')
        ro = -(re+rc)
        print(f'ro = {ro}')
        ke = float(lines[7])
        print(f'ke = {ke}')
        
        # Carico i file di ricerca degli zeri e interpolazione
        # x = r, y = r*
        folder_name = resolve_existing_path("TORTOISE_INVERSION_OUTPUT", require_dir=True)
        if rootFinding: x1, y1 = load_data(f"{folder_name}/output_1_{M}_{L}_{hrad}_{hstar}_.txt")
        x2, y2 = load_data(f"{folder_name}/output_2_{M}_{L}_{hrad}_{hstar}_.txt")
        x3, y3 = load_data(f"{folder_name}/output_3_{M}_{L}_{hrad}_{hstar}_.txt")

        # White line print
        stampa_riga_bianca()
        
        # First we define the metric parameters
        ignore_warnings()
     
        # We initialize the metric grr**-1 function
        global f
        global fl
        f = 1 - 2 * M / r - L / 3 * r ** 2
        f1d = diff(f,r)
        f2d = diff(f1d,r)
        f3d = diff(f2d,r)
        f4d = diff(f3d,r)
        f5d = diff(f4d,r)
        fl = lambdify(r, f, 'numpy')

        """
        ###################################################################
        Calcolo il potenziale dispari dei QNMs SdS
        ###################################################################
        """
        
        # Definisco il potenziale dispari
        s = 2 # Perturbazione gravitazionale
        
        global V0eff
        global V1eff
        global V2eff
        global V2effStar
        global V3eff
        global V3effStar
        global V4eff
        global V4effStar
        global V5eff
        global V5effStar
        global V6eff
        global V6effStar
        # - 2 / 3 * L * r ** 2
        # 0-th derivative
        V = f / r ** 2 * (l * (l + 1) + 2 * M / r * (1 - s ** 2))
        V0eff = lambdify((r, l), V, 'numpy')
        # 1st derivative
        V1d = diff(V,r)
        V1eff = lambdify((r, l), V1d, 'numpy')
        V1dStar = (V1d * f)
        V1effStar = lambdify((r, l), V1dStar, 'numpy')
        # 2nd derivatives
        V2d = diff(V1d,r)
        V2eff = lambdify((r, l), V2d, 'numpy')
        V2dStar = f * (f * V2d
                       + V1d * f1d)
        V2effStar = lambdify((r, l), V2dStar, 'numpy')
        # 3rd derivative
        V3d = diff(V2d,r)
        V3eff = lambdify((r, l), V3d, 'numpy')
        V3dStar = f * (V3d * f ** 2
                       + f * (3 * V2d * f1d + V1d * f2d)
                       + V1d * f1d ** 2)
        V3effStar = lambdify((r, l), V3dStar, 'numpy')
        # 4th derivative
        V4d = diff(V3d,r)
        V4eff = lambdify((r, l), V4d, 'numpy')
        V4dStar = f * (V4d * f ** 3
                       + f ** 2 * (6 * V3d * f1d + 4 * V2d * f2d + V1d * f3d)
                       + f * (7 * V2d * f1d ** 2 + 4 * V1d * f1d * f2d)
                       + V1d * f1d ** 3)
        V4effStar = lambdify((r, l), V4dStar, 'numpy')
        # 5th derivative
        V5d = diff(V4d,r)
        V5eff = lambdify((r, l), V5d, 'numpy')
        V5dStar = f * (V5d * f ** 4
                       + f ** 3 * (10 * V4d * f1d + 10 * V3d * f2d
                                   + 5 * V2d * f3d + V1d * f4d)
                       + f ** 2 * (25 * V3d * f1d ** 2 + 30 * V2d * f1d * f2d
                                   + 7 * V1d * f1d * f3d + 4 * V1d * f2d ** 2)
                       + f * (15 * V2d * f1d ** 3 + 11 * V1d * f1d ** 2 * f2d)
                       + V1d * f1d ** 4) 
        V5effStar = lambdify((r, l), V5dStar, 'numpy')
        # 6th derivative
        V6d = diff(V5d,r)
        V6eff = lambdify((r, l), V6d, 'numpy')
        V6dStar = f * (V6d * f ** 5
                       + f ** 4 * (15 * f1d * V5d + 20 * f2d * V4d + 15 * f3d * V3d
                                   + 6 * f4d * V2d + f5d * V1d)
                       + f ** 3 * (65 * f1d ** 2 * V4d + 120 * f1d * f2d * V3d
                                   + 57 * f1d * f3d * V2d + 34 * f2d ** 2 * V2d
                                   + 15 * f2d * f3d * V1d + 11 * f1d * f4d * V1d)
                       + f ** 2 * (90 * f1d ** 3 * V3d + 146 * f1d ** 2 * f2d * V2d
                                   + 32 * f1d ** 2 * f3d * V1d + 34 * f1d * f2d ** 2 * V1d)
                       + f * (31 * f1d ** 4 * V2d + 26 * f1d ** 3 * f2d * V1d)
                       + V1d * f1d ** 5)  
        V6effStar = lambdify((r, l), V6dStar, 'numpy')
        
        r0, defect = maximum_potential(V1eff, V2eff, V3eff, 2, re, rc, M)
        try:
            V0, V2, V3, V4, V5, V6 = interpolatePotential(V0eff, 2, V1eff, V2eff, V3eff, re, rc, M, x3, y3)
        except Exception as e:
            print(f'Exception \'{e}\' occurred.')
            # Fallback to analytic star derivatives at the potential peak.
            V2 = V2effStar(r0, 2)
            V3 = V3effStar(r0, 2)
            V4 = V4effStar(r0, 2)
            V5 = V5effStar(r0, 2)
            V6 = V6effStar(r0, 2)
        # Mostro alcuni valori dei potenziali in prossimità del massimo
        cifreSig = 5
        
        print(f'V2eff({r0:.{cifreSig}g},l=2) = {V2eff(r0,2):.{cifreSig}g}, V2effStar({r0:.{cifreSig}g},l=2) = {V2effStar(r0,2):.{cifreSig}g}, V2effInt({r0:.{cifreSig}g},l=2) = {V2:.{cifreSig}g}')
        print(f'V3eff({r0:.{cifreSig}g},l=2) = {V3eff(r0,2):.{cifreSig}g}, V3effStar({r0:.{cifreSig}g},l=2) = {V3effStar(r0,2):.{cifreSig}g}, V3effInt({r0:.{cifreSig}g},l=2) = {V3:.{cifreSig}g}')
        print(f'V4eff({r0:.{cifreSig}g},l=2) = {V4eff(r0,2):.{cifreSig}g}, V4effStar({r0:.{cifreSig}g},l=2) = {V4effStar(r0,2):.{cifreSig}g}, V4effInt({r0:.{cifreSig}g},l=2) = {V4:.{cifreSig}g}')
        print(f'V5eff({r0:.{cifreSig}g},l=2) = {V5eff(r0,2):.{cifreSig}g}, V5effStar({r0:.{cifreSig}g},l=2) = {V5effStar(r0,2):.{cifreSig}g}, V5effInt({r0:.{cifreSig}g},l=2) = {V5:.{cifreSig}g}')
        print(f'V6eff({r0:.{cifreSig}g},l=2) = {V6eff(r0,2):.{cifreSig}g}, V6effStar({r0:.{cifreSig}g},l=2) = {V6effStar(r0,2):.{cifreSig}g}, V6effInt({r0:.{cifreSig}g},l=2) = {V6:.{cifreSig}g}')
        
        """
        ###################################################################
        Calcolo il potenziale pari dei QNMs SdS
        ###################################################################
        """

        global V0effE

        c = 1 / 2 * (l + 2) * (l - 1)
        beta = 6*M
        
        Ve = (2 * f * (9 * M ** 3 + 9 * M ** 2 * c * r + 3 * c ** 2 * M * r ** 2
                       + c ** 2 * (c + 1) * r ** 3 - 3 * M ** 2 * L * r ** 3)
              / r ** 3 / (c * r + 3 * M) ** 2)
        Ve = expand(Ve)
        V0effE = lambdify((r, l), Ve, 'numpy')

        """
        ###################################################################
        Calcolo le frequenze a partire dal potenziale dispari
        ###################################################################
        """
        # Asymptotic limits analysis
        calcolaFrequenze = chiedi_conferma(f"Do you wish to compute sampled QNMS {omegaGreek}? Y / N : ")
        if calcolaFrequenze: 
            while calcolaFrequenze:
                try:
                    nExp = chiedi_numero_int_interval(f"What overtone number n do you pick? (n >= 0) : ",
                                                      limite_inferiore=float(0), limite_superiore=float(f"inf"))
                    lExp = chiedi_numero_int_interval(f"What multipole number l do you pick? (l >= 2) : ",
                                                      limite_inferiore=float(2), limite_superiore=float(f"inf"))
                    r0, defect = maximum_potential(V1eff, V2eff, V3eff, lExp, re, rc, M)
                    print(f'III order WKB approximation results:')
                    for i in range(nExp,nExp+3):
                        omega, omega2 = frequency(i,lExp)
                        print(f'{omegaGreek}(n={i},l={lExp}) = {omega}\n{omegaGreek}(n={i},l={lExp})**2 = {omega2}')
                    print(f'High multipole number l limit results:')
                    for i in range(nExp,nExp+3):
                        omega, omega2 = omega_high_l_nl(i,lExp)
                        print(f'{omegaGreek}(n={i},l={lExp}) = {omega}\n{omegaGreek}(n={i},l={lExp})**2 = {omega2}')
                    print(f'High overtone number n limit results:')
                    for i in range(nExp,nExp+3):
                        omega, omega2 = omega_high_n_nl(i,lExp)
                        print(f'{omegaGreek}(n={i},l={lExp}) = {omega}\n{omegaGreek}(n={i},l={lExp})**2 = {omega2}')
                    print(f'Frobenius/Leaver-like cross-check frequencies:')
                    for i in range(nExp, nExp + 3):
                        omega_wkb, _ = frequency(i, lExp)
                        omega_frob, omega2_frob = omega_frobenius(re, rc, ro, M, lExp, i, s)
                        rel_gap = abs(omega_frob - omega_wkb) / max(abs(omega_wkb), 1e-15)
                        print(
                            f'{omegaGreek}(n={i},l={lExp})_Frob = {omega_frob}\n'
                            f'{omegaGreek}(n={i},l={lExp})_Frob**2 = {omega2_frob}\n'
                            f'Relative gap vs WKB = {rel_gap}'
                        )
                    # White line print
                    stampa_riga_bianca()
                    # Chiedo se vuole riplottare
                    calcolaFrequenze = chiedi_conferma(f"Confirm whether you'd like to compute again sampled {omegaGreek}. Y / N : ")
                except Exception as e:
                    print(f'Exception \'{e}\' occurred.')
                    # White line print
                    stampa_riga_bianca()
                    calcolaFrequenze = chiedi_conferma(f"Confirm whether you'd like to compute again sampled {omegaGreek}. Y / N : ")

        # White line print
        stampa_riga_bianca()

        # Plotto le frequenze generate da WKB III
        mostraFrequenze = chiedi_conferma(f"Do you wish to plot sampled QNMS {omegaGreek}? Y / N : ")
        if mostraFrequenze: 
            while mostraFrequenze:
                try:
                    numN = chiedi_numero_int_interval(f"Up to which overtone number n you want to plot {omegaGreek}? (n > 0) : ",
                                                      limite_inferiore=float(0), limite_superiore=float(f"inf"))
                        
                    numL = chiedi_numero_int_interval(f"Up to which multipole number l you want to plot {omegaGreek}? (l >= 2) : ",
                                                      limite_inferiore=float(2), limite_superiore=float(f"inf"))
                        
                    freqReal = np.zeros((numN,numL))
                    freqImag = np.zeros((numN,numL))
                    for i in range(numN):
                        for j in range(2,numL):
                            omega, omega2 = frequency(i,j)
                            
                            freqReal[i,j] = np.real(omega)
                            freqImag[i,j] = np.imag(omega)

                    save_folder = normalize_output_folder('QNMS_FREQUENCY_PLOT', create=True)

                    for i in range(numN):
                        plt.scatter(freqReal[i,2:], freqImag[i,2:], label=f'{omegaGreek}(n={i})')
                    plt.xlabel(f'Re({omegaGreek})')
                    plt.ylabel(f'Im({omegaGreek})')
                    #plt.ylim(-.03,-.0015)
                    plt.title(f"{omegaGreek} plot sorted for n")
                    plt.legend()
                    safe_tight_layout()
                    plt.grid(True)
                    plt.savefig(f'{save_folder}/plotFreqN_{M}_{L}_{hstar}_{numN}_{numL}_.png')
                    print(f'File saved in {save_folder}/{save_folder}/plotFreq_{M}_{L}_{hstar}_{numN}_{numL}_.png')
                    show_or_close()

                    for j in range(2,numL):
                        plt.scatter(freqReal[:,j], freqImag[:,j], label=f'{omegaGreek}(l={j})')
                    plt.xlabel(f'Re({omegaGreek})')
                    plt.ylabel(f'Im({omegaGreek})')
                    #plt.ylim(-.03,-.0015)
                    plt.title(f"{omegaGreek} plot sorted for l")
                    plt.legend()
                    safe_tight_layout()
                    plt.grid(True)
                    plt.savefig(f'{save_folder}/plotFreqL_{M}_{L}_{hstar}_{numN}_{numL}_.png')
                    print(f'File saved in {save_folder}/{save_folder}/plotFreq_{M}_{L}_{hstar}_{numN}_{numL}_.png')
                    show_or_close()
                    
                    # White line print
                    stampa_riga_bianca()
                    # Chiedo se vuole riplottare
                    mostraFrequenze = chiedi_conferma(f"Confirm whether you'd like to plot again sampled {omegaGreek}. Y / N : ")
                except Exception as e:
                    print(f'Exception \'{e}\' occurred.')
                    # White line print
                    stampa_riga_bianca()
                    mostraFrequenze = chiedi_conferma(f"Confirm whether you'd like to plot again sampled {omegaGreek}. Y / N : ")

        # White line print
        stampa_riga_bianca()

        """
        ###################################################################
        Cross-check automatico tra metodi indipendenti
        ###################################################################
        """
        cross_check = chiedi_conferma(
            "Do you wish to run automated method cross-check (WKB/Frobenius/time-domain)? Y / N : "
        )
        while cross_check:
            try:
                n_check = chiedi_numero_int_interval(
                    "Pick overtone number n for cross-check (n >= 0) : ",
                    limite_inferiore=float(0), limite_superiore=float("inf")
                )
                l_check = chiedi_numero_int_interval(
                    "Pick multipole number l for cross-check (l >= 2) : ",
                    limite_inferiore=float(2), limite_superiore=float("inf")
                )
                include_td = chiedi_conferma(
                    "Include time-domain estimate in cross-check? Y / N : "
                )
                k_check = 5
                p_prony = 6
                max_points = 1800
                if include_td:
                    k_check = chiedi_numero_int_interval(
                        "Pick spaceTimeK for time-domain cross-check (K >= 1) : ",
                        limite_inferiore=float(1), limite_superiore=float("inf")
                    )
                    p_prony = chiedi_numero_int_interval(
                        "Pick Prony order p (2 <= p <= 12) : ",
                        limite_inferiore=float(2), limite_superiore=float(12)
                    )
                    max_points = chiedi_numero_int_interval(
                        "Pick max spatial points for PDE downsampling (200 <= n <= 6000) : ",
                        limite_inferiore=float(200), limite_superiore=float(6000)
                    )

                run_method_cross_check(
                    n_check, l_check, x3, y3,
                    include_time_domain=include_td,
                    space_time_k=k_check,
                    p_prony=p_prony,
                    max_points=max_points
                )
                stampa_riga_bianca()
                cross_check = chiedi_conferma(
                    "Confirm whether you'd like to run another method cross-check. Y / N : "
                )
            except Exception as e:
                print(f'Exception \'{e}\' occurred.')
                traceback.print_exc()
                stampa_riga_bianca()
                cross_check = chiedi_conferma(
                    "Confirm whether you'd like to run another method cross-check. Y / N : "
                )

        # White line print
        stampa_riga_bianca()

        """
        ###################################################################
        Analisi convergenza/errori su hrad, hstar e spaceTimeK
        ###################################################################
        """
        run_convergence = chiedi_conferma(
            "Do you wish to run convergence/error analysis (hrad/hstar/spaceTimeK)? Y / N : "
        )
        while run_convergence:
            try:
                n_conv = chiedi_numero_int_interval(
                    "Pick overtone number n for convergence analysis (n >= 0) : ",
                    limite_inferiore=float(0), limite_superiore=float("inf")
                )
                l_conv = chiedi_numero_int_interval(
                    "Pick multipole number l for convergence analysis (l >= 2) : ",
                    limite_inferiore=float(2), limite_superiore=float("inf")
                )
                max_variants = chiedi_numero_int_interval(
                    "How many inversion variants do you want to sample? (2 <= n <= 12) : ",
                    limite_inferiore=float(2), limite_superiore=float(12)
                )
                k_min = chiedi_numero_int_interval(
                    "Set minimum spaceTimeK for sweep (K >= 1) : ",
                    limite_inferiore=float(1), limite_superiore=float("inf")
                )
                k_max = chiedi_numero_int_interval(
                    f"Set maximum spaceTimeK for sweep (K >= {k_min}) : ",
                    limite_inferiore=float(k_min), limite_superiore=float("inf")
                )
                k_step = chiedi_numero_int_interval(
                    "Set increment for spaceTimeK sweep (step >= 1) : ",
                    limite_inferiore=float(1), limite_superiore=float("inf")
                )
                space_time_ks = list(range(k_min, k_max + 1, k_step))
                if len(space_time_ks) > 7:
                    idx = np.linspace(0, len(space_time_ks) - 1, 7, dtype=int)
                    idx = np.unique(idx)
                    space_time_ks = [space_time_ks[i] for i in idx]
                    print(f"SpaceTimeK sweep reduced to representative values: {space_time_ks}")

                p_prony = chiedi_numero_int_interval(
                    "Pick Prony order p for time-domain extraction (2 <= p <= 12) : ",
                    limite_inferiore=float(2), limite_superiore=float(12)
                )
                max_points = chiedi_numero_int_interval(
                    "Pick max spatial points for PDE downsampling (200 <= n <= 6000) : ",
                    limite_inferiore=float(200), limite_superiore=float(6000)
                )

                run_convergence_error_analysis(
                    n_conv, l_conv, space_time_ks,
                    p_prony=p_prony,
                    max_variants=max_variants,
                    max_points=max_points,
                    base_r=x3,
                    base_rstar=y3
                )
                stampa_riga_bianca()
                run_convergence = chiedi_conferma(
                    "Confirm whether you'd like to run another convergence/error analysis. Y / N : "
                )
            except Exception as e:
                print(f'Exception \'{e}\' occurred.')
                traceback.print_exc()
                stampa_riga_bianca()
                run_convergence = chiedi_conferma(
                    "Confirm whether you'd like to run another convergence/error analysis. Y / N : "
                )

        """
        ###################################################################
        Testo il calcolatore della funzione asintotica r -> re con Frobenius
        ###################################################################
        """
        """
        print(f'Frobenius asymptotic correction')
        nExp = 0
        lExp = 10
        omega, omega2 = frequency(nExp,lExp)
        print(f'{omegaGreek}(n={nExp},l={lExp}) = {omega}\n{omegaGreek}(n={nExp},l={lExp})**2 = {omega2}')
        indexStar = indice_valore_piu_vicino(x3, r0)
        r0Star = y3[indexStar]
        ufunc, u = funzione_u(re,rc,ro,M,lExp,omega,s)
        #print(u)
        #print(ufunc(x3))
        uYs = np.real(ufunc(np.reciprocal(x3)))
        #print(f'uYs = {uYs}')
        ys = V0eff(x3,lExp)
        inWaveYs = np.exp(-1j * omega * y3)

        # White line print
        stampa_riga_bianca()
        """
        """
        ###################################################################
        Testo la soluzione della PDE con Gundlach, Price e Pullin
        ###################################################################
        """
        lPDE = 2
        print(f'Gundlach, Price & Pullin PDE solution')
        hStar = hstar
        cercaPDE = chiedi_conferma("Do you wish to look for the PDE solution? Y / N : ")
        if cercaPDE:
            # Estraggo la soluzione se precedentemente calcolata
            spaceTimeK = 5
            trovato, phi = trova_file_e_estrai_matrice(M, L, lPDE, hStar, spaceTimeK)
            
            if trovato:
                print(f'Loading preexisting parameters.')
                # Se trova la matrice cerca anche la coordinata t
                trovatoT, t = trova_file_e_estrai_t(M, L, lPDE, hStar, spaceTimeK)
                rStarUV, rUV, rStar0 = symmetricRStar(x3,y3)
                dim = int((len(t)+1) / 2)
                if (not trovatoT) or dim < 10 or min(phi.shape) < 5:
                    print(
                        "Preexisting PDE cache is too small/invalid for analysis; "
                        "recomputing PDE solution with current grid settings."
                    )
                    trovato = False
            if not trovato:
                print(f'PDE solution calculation.')
                i0, phi, t, dim, rStarUV, rUV = pde_solution(x3,y3,lPDE,spaceTimeK)
                salva_t_coordinate(t, hStar, M, L, lPDE, spaceTimeK)
                salva_soluzione_pde(phi, hstar, M, L, lPDE, spaceTimeK)

            ### Implementare l'algoritmo while per poter ripetere i plot più volte, se si vuole
            riplotta = chiedi_conferma("Do you wish to plot the PDE solution? Y / N : ")
            while riplotta:
                try:
                    # White line print
                    stampa_riga_bianca()
                    
                    # Si stampa il profilo SpazioTemporale
                    spaceTimePlot(phi,PsiGreek,frase=f'{M}_{L}_{lPDE}_{hStar}_{spaceTimeK}')

                    # White line print
                    stampa_riga_bianca()
                    
                    # Si stampano i profili temporali
                    timePlot(phi,PsiGreek,omegaGreek,dim,t,rUV,fl,hStar,spaceTimeK,
                             frase=f'{M}_{L}_{lPDE}_{hStar}_{spaceTimeK}')
                    
                    # White line print
                    stampa_riga_bianca()

                    # Si stampano i profili spaziali
                    spacePlot(phi,PsiGreek,omegaGreek,dim,t,rStarUV,rUV,fl,hStar,spaceTimeK,
                              frase=f'{M}_{L}_{lPDE}_{hStar}_{spaceTimeK}')
                    
                    # White line print
                    stampa_riga_bianca()
                    riplotta = chiedi_conferma(f"Do you wish to plot again the PDE solution? Y / N : ")
                except Exception as e:
                    print(f'Exception \'{e}\' occurred.')
                    traceback.print_exc()
                    
                    # White line print
                    stampa_riga_bianca()
                        
                    riplotta = chiedi_conferma(f"Do you wish to plot again the PDE solution? Y / N : ")

        # White line print
        stampa_riga_bianca()
        
        """
        ####################################
        Plotto il potenziale in funzione di r*
        ####################################
        """
        riplotta = chiedi_conferma("Do you wish to plot V vs r*? Y / N : ")
        while riplotta:
            lplot = rl.chiedi_numero_int("Pick integer l >= 2 : ")
            ys = V0eff(x2,lplot)
            xs = y2
            fig, ax = plt.subplots(figsize=(16, 9))
            plt.plot(xs, ys, label='y = Gravitational Perturbation Potential')
            plt.xlabel('r*')
            plt.ylabel('Veff')
            plt.title(f"Effective potential plot for  l = {lplot} vs r*")
            plt.legend()
            safe_tight_layout()
            plt.grid(True)
            show_or_close()
            riplotta = chiedi_conferma("Do you wish to plot again V vs r*? Y / N : ")

        # White line print
        stampa_riga_bianca()
        
        """
        ####################################
        Plotto il potenziale in funzione di r
        ####################################
        """
        riplotta = chiedi_conferma("Do you wish to plot V vs r? Y / N : ")
        while riplotta:
            lplot = rl.chiedi_numero_int("Pick integer l >= 2 : ")
            ys = V0eff(x2,lplot)
            xs = x2
            fig, ax = plt.subplots(figsize=(16, 9))
            plt.plot(xs, ys, label='y = Gravitational Perturbation Potential')
            plt.xlabel('r')
            plt.ylabel('Veff')
            plt.title(f"Effective potential plot for  l = {lplot} vs r")
            plt.legend()
            safe_tight_layout()
            plt.grid(True)
            show_or_close()
            riplotta = chiedi_conferma("Do you wish to plot V vs r? Y / N : ")

        # White line print
        stampa_riga_bianca()
        
        # Imposto la definizione di derivata seconda centrale pointwise
        # f2 = (func(r* + h) - 2 * func(r*) + func(r* - h)) / (4 * h ** 2)
        # La somma scorre tra 0 e + inf. Si deve ricorrere ad un'approssimazione.
        # LMIN, LMAX sono gli estremi di tale sommatoria.
        # Conoscendo la formula asintotica per grandi l di omega_nl, la applico, sapendo che i contributi più importanti sono per alti l
        """
        ####################################
        Soluzione dell'ODE con Runge-Kutta IV
        ####################################
        """
        print(f'Soluzione dell\'ODE con Runge-Kutta IV')
        riplotta = chiedi_conferma(f"State whether you'd like to compute and show {PsiGreek} spacelike profile. Y / N : ")
        while riplotta:
            try:
                #LMIN = chiedi_input_time('Set LMIN value for series truncation : ', 10)
                #LMAX = chiedi_input_time('Set LMAX value for series truncation : ', 100)
                nRK = chiedi_numero_int_interval(f"What overtone number n do you pick? (n>=0) : ",
                                                   limite_inferiore=float(0), limite_superiore=float("inf"))
                lRK = chiedi_numero_int_interval(f"What multipole number l do you pick? (l>=2) : ",
                                                   limite_inferiore=float(2), limite_superiore=float("inf"))
                     
                # Devo ricavare il risultato odd e il risultato even e plottarli assieme
                

                # Special computation with RKII + RKIV
                ym3oddS = rk4Special(x3,y3,lRK,nRK,frobCorr=True)
                ym3evenS = rk4Special(x3,y3,lRK,nRK,even=True,frobCorr=True)
                save_plot_with_progressive(x3, y3, ym3oddS, 'r', 'Re(R)',
                                           f'QNMs RKIV odd solution with II order r* inversion (n={nRK}, l={lRK})',
                                           special=True)
                save_plot_with_progressive(x3, y3, ym3evenS, 'r', 'Re(R)',
                                           f'QNMs RKIV even solution with II order r* inversion (n={nRK}, l={lRK})',
                                           special=True)
                save_plot_with_progressive(x3, y3, ym3oddS, 'r', 'Re(R)',
                                           f'QNMs RKIV odd solution with II order r* inversion (n={nRK}, l={lRK})',
                                           plotTortoise=True,special=True)
                save_plot_with_progressive(x3, y3, ym3evenS, 'r', 'Re(R)',
                                           f'QNMs RKIV even solution with II order r* inversion (n={nRK}, l={lRK})',
                                           plotTortoise=True,special=True)
                save_plot_with_progressive(x3, y3, ym3oddS, 'r', 'log|Re(R)|',
                                           f'QNMs RKIV odd solution with II order r* inversion (n={nRK}, l={lRK})',
                                           semiLog=True,special=True)
                save_plot_with_progressive(x3, y3, ym3evenS, 'r', 'log|Re(R)|',
                                           f'QNMs RKIV even solution with II order r* inversion (n={nRK}, l={lRK})',
                                           semiLog=True,special=True)
                save_plot_with_progressive(x3, y3, ym3oddS, 'r', 'log|Re(R)|',
                                           f'QNMs RKIV odd solution with II order r* inversion (n={nRK}, l={lRK})',
                                           plotTortoise=True,semiLog=True,special=True)
                save_plot_with_progressive(x3, y3, ym3evenS, 'r', 'log|Re(R)|',
                                           f'QNMs RKIV even solution with II order r* inversion (n={nRK}, l={lRK})',
                                           plotTortoise=True,semiLog=True,special=True)  

                # Ora devo computare il modo pari a partire dal modo dispari e viceversa Special
                nFr = nRK
                lFr = lRK
                l = lFr
                c = 1 / 2 * (l + 2) * (l - 1)
                dK = 4 * c * (c + 1)
                
                fTilde = f / (2 * r * (c * r + 3 * M))
                
                omega, omega2 = frequency(nFr,lFr)
                omega_sym = Float(float(np.real(omega))) + I * Float(float(np.imag(omega)))
                
                oddTerm1 = ((dK + 2 * beta ** 2 * fTilde) / (dK + 2j * omega_sym * beta))
                oddTerm2 = 2 * beta / (dK + 2j * omega_sym * beta)
                oddTerm1l = lambdify(r,oddTerm1,'numpy')
                oddTerm2l = lambdify(r,oddTerm2,'numpy')
                
                evenTerm1 = (dK + 2 * beta ** 2 * fTilde) / (dK - 2j * omega_sym * beta)
                evenTerm2 = 2 * beta / (dK - 2j * omega_sym * beta)
                evenTerm1l = lambdify(r,evenTerm1,'numpy')
                evenTerm2l = lambdify(r,evenTerm2,'numpy')
                
                oddTerm1Vals = oddTerm1l(x3)
                oddTerm2Vals = oddTerm2l(x3)
                evenTerm1Vals = evenTerm1l(x3)
                evenTerm2Vals = evenTerm2l(x3)

                ym3oddEvenS = oddTerm1Vals * ym3oddS[0,:] + oddTerm2Vals * ym3oddS[1,:]
                ym3evenOddS = evenTerm1Vals * ym3evenS[0,:] - evenTerm2Vals * ym3evenS[1,:]

                save_folder = normalize_output_folder('QNMS_PDE_SPACELIKE_PROFILE', create=True)

                params = f'{M}_{L}_{nRK}_{lRK}_{hStar}'
                y3Plot = to_real_float_array(y3)
                x3Plot = to_real_float_array(x3)
                oddPlot = to_real_float_array(ym3oddS[0, :])
                evenPlot = to_real_float_array(ym3evenS[0, :])
                oddEvenPlot = to_real_float_array(ym3oddEvenS)
                evenOddPlot = to_real_float_array(ym3evenOddS)
                oddDefect = safe_relative_defect(oddPlot, evenOddPlot)
                evenDefect = safe_relative_defect(evenPlot, oddEvenPlot)

                yOddX, yOddY = finite_xy(y3Plot, oddPlot, "Odd profile in r*")
                yEvenOddX, yEvenOddY = finite_xy(y3Plot, evenOddPlot, "Transformed odd profile in r*")
                plt.plot(yOddX, yOddY, label='y = Odd')
                plt.plot(yEvenOddX, yEvenOddY, label='y = Transformed odd')
                plt.xlabel('r*')
                plt.ylabel('Re odd QNM')
                plt.title(f"Odd vs transformed odd (n={nRK},l={lRK})")
                plt.legend()
                safe_tight_layout()
                plt.grid(True)
                plt.savefig(f'{save_folder}/plotStarS_odd_{params}_.png')
                show_or_close()
                print(f'File saved in {save_folder}/plotStarS_odd_{params}_.png')
                
                yEvenX, yEvenY = finite_xy(y3Plot, evenPlot, "Even profile in r*")
                yOddEvenX, yOddEvenY = finite_xy(y3Plot, oddEvenPlot, "Transformed even profile in r*")
                plt.plot(yEvenX, yEvenY, label='y = Even')
                plt.plot(yOddEvenX, yOddEvenY, label='y = Transformed even')
                plt.xlabel('r*')
                plt.ylabel('Re even QNM')
                plt.title(f"Even vs transformed even (n={nRK},l={lRK})")
                plt.legend()
                safe_tight_layout()
                plt.grid(True)
                plt.savefig(f'{save_folder}/plotStarS_even_{params}_.png')
                show_or_close()
                print(f'File saved in {save_folder}/plotStarS_even_{params}_.png')

                yOddDefX, yOddDefY = finite_xy(y3Plot, y_semilogy(oddDefect), "Odd defect in r*")
                plt.semilogy(yOddDefX, yOddDefY, label='y = Odd defect')
                plt.xlabel('r*')
                plt.ylabel('Odd defect')
                plt.title(f"Odd parity Darboux transformation relative defect (n={nRK},l={lRK})")
                plt.legend()
                safe_tight_layout()
                plt.grid(True)
                plt.savefig(f'{save_folder}/semilogyStarS_oddDefect_{params}_.png')
                show_or_close()
                print(f'File saved in {save_folder}/semilogyStarS_oddDefect_{params}_.png')

                yEvenDefX, yEvenDefY = finite_xy(y3Plot, y_semilogy(evenDefect), "Even defect in r*")
                plt.semilogy(yEvenDefX, yEvenDefY, label='y = Even defect')
                plt.xlabel('r*')
                plt.ylabel('Even defect')
                plt.title(f"Even parity Darboux transformation relative defect (n={nRK},l={lRK})")
                plt.legend()
                safe_tight_layout()
                plt.grid(True)
                plt.savefig(f'{save_folder}/semilogyStarS_evenDefect_{params}_.png')
                show_or_close()
                print(f'File saved in {save_folder}/semilogyStarS_evenDefect_{params}_.png')

                xOddX, xOddY = finite_xy(x3Plot, oddPlot, "Odd profile in r")
                xEvenOddX, xEvenOddY = finite_xy(x3Plot, evenOddPlot, "Transformed odd profile in r")
                plt.plot(xOddX, xOddY, label='y = Odd')
                plt.plot(xEvenOddX, xEvenOddY, label='y = Transformed odd')
                plt.xlabel('r')
                plt.ylabel('Re QNM')
                plt.title(f"Odd vs transformed odd (n={nRK},l={lRK})")
                plt.legend()
                safe_tight_layout()
                plt.grid(True)
                plt.savefig(f'{save_folder}/plotS_odd_{params}_.png')
                show_or_close()
                print(f'File saved in {save_folder}/plotS_odd_{params}_.png')
                
                xEvenX, xEvenY = finite_xy(x3Plot, evenPlot, "Even profile in r")
                xOddEvenX, xOddEvenY = finite_xy(x3Plot, oddEvenPlot, "Transformed even profile in r")
                plt.plot(xEvenX, xEvenY, label='y = Even')
                plt.plot(xOddEvenX, xOddEvenY, label='y = Transformed even')
                plt.xlabel('r')
                plt.ylabel('Re QNM')
                plt.title(f"Even vs transformed even (n={nRK},l={lRK})")
                plt.legend()
                safe_tight_layout()
                plt.grid(True)
                plt.savefig(f'{save_folder}/plotS_even_{params}_.png')
                show_or_close()
                print(f'File saved in {save_folder}/plotS_even_{params}_.png')

                xOddDefX, xOddDefY = finite_xy(x3Plot, y_semilogy(oddDefect), "Odd defect in r")
                plt.semilogy(xOddDefX, xOddDefY, label='y = Odd defect')
                plt.xlabel('r')
                plt.ylabel('Odd defect')
                plt.title(f"Odd parity Darboux transformation relative defect (n={nRK},l={lRK})")
                plt.legend()
                safe_tight_layout()
                plt.grid(True)
                plt.savefig(f'{save_folder}/semilogyS_oddDefect_{params}_.png')
                show_or_close()
                print(f'File saved in {save_folder}/semilogyS_oddDefect_{params}_.png')

                xEvenDefX, xEvenDefY = finite_xy(x3Plot, y_semilogy(evenDefect), "Even defect in r")
                plt.semilogy(xEvenDefX, xEvenDefY, label='y = Even defect')
                plt.xlabel('r')
                plt.ylabel('Even defect')
                plt.title(f"Even parity Darboux transformation relative defect (n={nRK},l={lRK})")
                plt.legend()
                safe_tight_layout()
                plt.grid(True)
                plt.savefig(f'{save_folder}/semilogyS_evenDefect_{params}_.png')
                show_or_close()
                print(f'File saved in {save_folder}/semilogyS_evenDefect_{params}_.png')

                # White line print
                stampa_riga_bianca()
                
                riplotta = chiedi_conferma(f"Confirm whether you'd like to compute and show {PsiGreek} spacelike profile. Y / N : ")
            except Exception as e:
                print(f'Exception \'{e}\' occurred.')
                traceback.print_exc()

                # White line print
                stampa_riga_bianca()
                    
                riplotta = chiedi_conferma(f"Confirm whether you'd like to compute and show {PsiGreek} spacelike profile. Y / N : ")

        # White line print
        stampa_riga_bianca()
        
    except KeyboardInterrupt:
        # Gestisce l'interruzione del programma con Ctrl+C
        print("\nProgram terminated.")


    
