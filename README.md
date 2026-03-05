# SdS-QNMs

Toolkit Python per lo studio dei Quasi-Normal Modes (QNMs) in spazio-tempo di Schwarzschild-de Sitter (SdS), con:

- inversione della coordinata tortoise `r*`
- soluzione della master equation (ODE)
- soluzione PDE in coordinate caratteristiche (Gundlach-Price-Pullin)
- confronto metodi (WKB / Frobenius-like / time-domain)
- analisi di convergenza su `hrad`, `hstar`, `spaceTimeK`

## Struttura principale

- `tortoise_inversion.py`: genera i dati `r <-> r*` e i parametri runtime.
- `QNMs_Master_Equation_Solution.py`: pipeline principale (potenziali, frequenze, PDE, profili).
- `qnms_utils.py`: utility numeriche, I/O PDE, fit frequenze, plotting.
- `rosignoli_lib.py`: validazione input e helper CLI.

## Requisiti

- Python 3.10+ (consigliato stesso ambiente con cui esegui già il progetto)
- Pacchetti: `numpy`, `sympy`, `matplotlib`, `tqdm`, `mpmath`, `rich`, `inputimeout`, `pypiexperiment`

## Installazione dipendenze

Installa i pacchetti da `requirements.txt`:

```bash
cd SdS-QNMs
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Se usi conda, puoi anche installare direttamente senza `venv`:

```bash
cd SdS-QNMs
pip install -r requirements.txt
```

## Esecuzione rapida

1. Genera inversione tortoise e parametri:

```bash
cd SdS-QNMs
python3 tortoise_inversion.py
```

2. Esegui la soluzione della master equation:

```bash
python3 QNMs_Master_Equation_Solution.py
```

## Variabili ambiente utili

- `QNMS_PDE_MAX_POINTS` (default: `6000`): limite punti usati per PDE dopo downsampling.

```bash
QNMS_PDE_MAX_POINTS=8000 python3 QNMs_Master_Equation_Solution.py
```

- `QNMS_SHOW_PLOTS`:
  - `0` (default): backend non interattivo, salva/chiude figure
  - `1`: mostra le finestre plot a schermo

```bash
QNMS_SHOW_PLOTS=1 python3 QNMs_Master_Equation_Solution.py
```

## Output generati

Le esecuzioni creano cartelle di output, tra cui:

- `TORTOISE_INVERSION_OUTPUT/`
- `TORTOISE_INVERSION_PLOTS/`
- `QNMS_PLOTS/`
- `QNMS_PDE_SOL/`, `QNMS_PDE_SOL_T/`
- `QNMS_PDE_TIME_PROFILE/`, `QNMS_PDE_SPACE_PROFILE/`, `QNMS_PDE_SPACELIKE_PROFILE/`
- `QNMS_METHOD_CROSSCHECK/`
- `QNMS_CONVERGENCE_ANALYSIS/`

## Note operative

- Se trovi solo un file `output_3_*`, l'analisi di convergenza su `hstar/hrad` viene salvata ma senza stima dell'ordine.
- I cache PDE piccoli/invalidi vengono scartati e ricalcolati automaticamente.
- Il caricamento PDE usa cache binaria `.npy` (piu veloce) con fallback `.txt`.
