from sympy import *
import sys
from rosignoli_lib import *
import mysql.connector
from config_secret import config
import getpass

def instaura_connessione():
    # Importa i valori secretati
    user = config['user']
    password = config['password']
    host = config['host']
    port = config['port']
    database = config['database']

    # Crea la connessione
    conn = mysql.connector.connect(user=user, password=password, host=host, port=port, database=database)
    return conn    

def seleziona_funzione_from_db():
    # Instaura la connessione globale
    conn = instaura_connessione()
    display_available_functions(conn)
    a, b, c, d, z, x = symbols('a b c d z x')
    codice = input("Inserisci il codice della funzione da selezionare: ")

    if codice.upper() == 'G':
        codice = add_function(conn)
        return None

    if codice.upper() == 'H':
        password = getpass.getpass("Inserisci la password per confermare l'eliminazione: ")
        if password != config['password']:
            print("Password errata. L'eliminazione non è stata confermata.")
            return
        codice = delete_chosen_function(conn)
        return None
    
    # Selezione funzione da DB
    result = get_function_by_codice(conn, codice)
    #print(result)
    if result:
        print(f"Funzione trovata: {result[1]}")
        f0 = sympify(result[1])
        f1 = diff(f0, z)
        f2 = diff(f1, z)
        f3 = diff(f2, z)
        
        f = lambdify(z, f0)
        df = lambdify(z, f1)
        ddf = lambdify(z, f2)
        dddf = lambdify(z, f3)
        
        return f, df, ddf, dddf, f0
    else:
        print("Funzione non trovata.")
        return None

if __name__ == "__main__":
    try:
        # Seleziona la funzione dal database
        selected_function = seleziona_funzione_from_db()


    except KeyboardInterrupt:
        # Gestisce l'interruzione del programma con Ctrl+C
        print("\nProgramma terminato.")
        

