# LIBRERIA INTERAMENTE SVILUPPATA DA IMAN ROSIGNOLI 18-08-1998 TERNI (TR)
def read_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return lines

def ignore_warnings():
    import warnings
    warnings.filterwarnings("ignore")

"""
#######################################################
Funzioni che accettano solo un determinato tipo di input.
#######################################################
"""

def chiedi_char(frase="Inserisci un carattere: "):
    while True:
        carattere = input(frase)
        if check_quit(carattere):
            print("Applicazione terminata.")
            quit()
        if len(carattere) == 1 and not carattere.isdigit(): # Controlla se il carattere è una lettera
            return carattere  # Restituisci il carattere se è valido
        else:
            print("Input non valido. Inserisci un solo carattere.")

def chiedi_index(frase="Inserisci un indice: "):
    while True:
        indice = input(frase)
        if check_quit(indice):
            print("Applicazione terminata.")
            quit()
        if len(indice) >= 1 and len(indice) <= 4:
            if indice[0].isalpha() and (len(indice) == 1 or indice[1:].isdigit()):
                return indice.upper()
        print("Input non valido. Inserisci un indice nel formato corretto (A, B, C1, D24).")

def chiedi_conferma(frase="Inserisci S o N: "):
    while True:
        user_input = input(frase)
        if check_quit(user_input):
            print("Applicazione terminata.")
            quit()
        if user_input.upper() == "S" or user_input.upper() == "Y":
            return True
        elif user_input.upper() == "N":
            return False
        else:
            print("Input non valido. Riprova.")

def chiedi_numero(frase="Inserisci l'input: "):
    while True:
        try:
            user_input = input(frase)
            if check_quit(user_input):
                print("Applicazione terminata.")
                quit()
            numero = float(user_input)
            return numero  # Restituisci il numero se è valido
        except ValueError:
            print("Input non valido. Riprova.")

def chiedi_numero_interval(frase="Inserisci un numero compreso tra {limite_inferiore} e {limite_superiore}: ",
                  limite_inferiore=float("-inf"), limite_superiore=float("inf")):
    while True:
        try:
            user_input = input(frase.format(limite_inferiore=limite_inferiore, limite_superiore=limite_superiore))
            if check_quit(user_input):
                print("Applicazione terminata.")
                quit()
            numero = float(user_input)
            if limite_inferiore <= numero <= limite_superiore:
                return numero  # Restituisci il numero se è compreso tra gli estremi
            else:
                print("Il numero inserito non è compreso tra gli estremi. Riprova.")
        except ValueError:
            print("Input non valido. Riprova.")

def chiedi_numero_int_interval(frase="Inserisci un numero intero compreso tra {limite_inferiore} e {limite_superiore}: ",
                  limite_inferiore=float("-inf"), limite_superiore=float("inf")):
    while True:
        try:
            user_input = input(frase.format(limite_inferiore=limite_inferiore, limite_superiore=limite_superiore))
            if check_quit(user_input):
                print("Applicazione terminata.")
                quit()
            numero = int(user_input)
            if limite_inferiore <= numero <= limite_superiore:
                return numero  # Restituisci il numero se è compreso tra gli estremi
            else:
                print("Il numero inserito non è compreso tra gli estremi. Riprova.")
        except ValueError:
            print("Input non valido. Riprova.")

def chiedi_numero_int(frase="Inserisci l'input: "):
    while True:
        try:
            user_input = input(frase)
            if check_quit(user_input):
                print("Applicazione terminata.")
                quit()
            numero = int(user_input)
            return numero  # Restituisci il numero se è valido
        except ValueError:
            print("Input non valido. Riprova.")

def chiedi_numero_pos(frase="Inserisci l'input: "):
    while True:
        try:
            user_input = input(frase)
            if check_quit(user_input):
                print("Applicazione terminata.")
                quit()
            numero = float(user_input)
            if numero > 0: # Controlla se il carattere è una lettera
                return numero  # Restituisci il carattere se è valido
            else:
                print("Input non valido. Inserisci un numero positivo.")
        except ValueError:
            print("Input non valido. Riprova.")

def chiedi_numero_int_pos(frase="Inserisci l'input: "):
    while True:
        try:
            user_input = input(frase)
            if check_quit(user_input):
                print("Applicazione terminata.")
                quit()
            numero = float(user_input)  # Accetta notazione scientifica
            if numero.is_integer() and numero > 0:
                return int(numero)
            else:
                print("Input non valido. Inserisci un numero intero positivo.")
        except ValueError:
            print("Input non valido. Riprova.")

def chiedi_numero_neg(frase="Inserisci l'input: "):
    while True:
        try:
            user_input = input(frase)
            if check_quit(user_input):
                print("Applicazione terminata.")
                quit()
            numero = float(user_input)
            if numero < 0: # Controlla se il carattere è una lettera
                return numero  # Restituisci il carattere se è valido
            else:
                print("Input non valido. Inserisci un numero positivo.")
        except ValueError:
            print("Input non valido. Riprova.")
            
def fattoriale(n, memo={}):
    if n in memo:
        return memo[n]
    elif n <= 1:
        return 1
    else:
        risultato = n * fattoriale(n-1, memo)
        memo[n] = risultato
        return int(risultato)

def letter_to_index(letter):
    if letter.isalpha():
        return ord(letter.upper()) - ord('A')
    else:
        raise ValueError("Input non valido. Deve essere una lettera.")

def index_to_letter(index):
    if 0 <= index < 26:
        return chr(index + ord('A'))
    else:
        raise ValueError("Input non valido. L'indice deve essere compreso tra 0 e 25.")

"""
#######################################################
Funzioni utili per Python-SQL.
#######################################################
"""

def custom_sort_key(item):
    code = item[0]
    letter_part = ''.join(filter(str.isalpha, code))  # Extract the letter part
    number_part = ''.join(filter(str.isdigit, code))  # Extract the number part
    
    if number_part:
        num_value = int(number_part)
    else:
        num_value = 0
        
    #print(item, num_value)
    index = letter_to_index(letter_part) + num_value * 26
    return index

def display_available_functions(conn):
    cursor = conn.cursor()
    query = "SELECT * FROM function_table"
    cursor.execute(query)
    
    functions = cursor.fetchall()
    
    # Sort the functions using the custom sort key
    sorted_functions = sorted(functions, key=custom_sort_key)
    
    print("Available Functions:")
    for row in sorted_functions:
        print(f"{row[0]}: {row[1]}")
    
    cursor.close()
    
def create_function(conn, codice, funzione):
    query = "INSERT INTO function_table (codice, funzione) VALUES (%s, %s)"
    values = (codice, funzione)
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()
    cursor.close()

def get_function_by_codice(conn, codice):
    query = "SELECT * FROM function_table WHERE codice = %s"
    values = (codice,)
    cursor = conn.cursor()
    cursor.execute(query, values)
    result = cursor.fetchone()
    cursor.close()
    return result

def update_function(conn, codice, nuova_funzione):
    query = "UPDATE function_table SET funzione = %s WHERE codice = %s"
    values = (nuova_funzione, codice)
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()
    cursor.close()

def delete_function(conn, codice):
    query = "DELETE FROM function_table WHERE codice = %s"
    values = (codice,)
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()
    cursor.close()

def add_function(conn):
    nuova_funzione = input("""Inserisci un'espressione per la funzione nella variabile z
f(z) = """)
    codice = get_next_function_code(conn)

    create_function(conn, codice, nuova_funzione)
    return codice
    #print(f"Funzione aggiunta con successo. Codice: {codice}")

def delete_chosen_function(conn):
    codice_da_cancellare = input("Inserisci il codice della funzione da cancellare: ")

    if codice_da_cancellare.upper() in ['G', 'H']:
        print("Impossibile cancellare le funzioni con i codici G e H.")
        return
    
    # Richiama la funzione di cancellazione
    delete_function(conn, codice_da_cancellare)

    # Chiudi la connessione
    conn.close()

def get_next_function_code(conn):
    cursor = conn.cursor()
    query = "SELECT * FROM function_table"
    cursor.execute(query)
    functions = cursor.fetchall()
    indexes = [custom_sort_key(item) for item in functions]
    primo_numero = primo_numero_mancante(indexes) # Non mi prende il primo numero mancante
    num = primo_numero // 26
    lettera = index_to_letter(primo_numero - num * 26)
    if num == 0:
        codice = f'{lettera}'
    else: 
        codice = f'{lettera}{num}'
    return codice

"""
#######################################################
Funzioni utili alla metaprogrammazione.
#######################################################
"""
  
def elemento_lista(lettera, lista):
    for elemento in lista:
        separatore = " : "
        indice_separatore = elemento.find(separatore)
        if indice_separatore != -1:
            lettera_elemento = elemento[:indice_separatore].strip().upper()
            espressione = elemento[indice_separatore + len(separatore):]
            if lettera_elemento == lettera.strip().upper():
                return espressione
    print("Nessuna espressione trovata per la lettera", lettera)

def elemento_lettera(title, lista):
    lettera = None
    for elemento in lista:
        lettera, nome = elemento.split(" : ")
        if nome == title:
            return lettera
            break
    print("Nessuna lettera trovata per l'espressione", title)

def check_esc():
    return keyboard.is_pressed('esc')

def check_quit(user_input):
    return "quit()" in user_input.lower()

def leggi_lista_da_file(nome_file, nome_lista = 'polinomi'):
    linea_dichiarazione = None

    with open(nome_file, "r") as file:
        for line in file:
            if nome_lista + " =" in line:
                linea_dichiarazione = line
                break

    elementi = []

    if linea_dichiarazione:
        # Rimuovi "frutti =" dalla linea di dichiarazione
        linea_senza_dichiarazione = linea_dichiarazione.replace(nome_lista + " =", "")

        # Rimuovi spazi bianchi e parentesi quadre
        linea_senza_dichiarazione = linea_senza_dichiarazione.strip().strip("[]")

        # Separa gli elementi utilizzando la virgola come delimitatore
        elementi = linea_senza_dichiarazione.split(",")

    # Rimuovi eventuali spazi bianchi dagli elementi
    elementi = [elemento.strip() for elemento in elementi]

    return elementi

def cancella_righe_lista(file_name, nome_lista = "polinomi"):
    with open(file_name, 'r+') as file:
        contenuto = file.readlines()
        file.seek(0)  # Torna all'inizio del file

        for linea in contenuto:
            if not linea.startswith("    " + nome_lista + " =") and not linea.startswith("    " + nome_lista + ".append("):
                file.write(linea)  # Scrivi solo le linee diverse da quelle da cancellare

        file.truncate()  # Riduci la dimensione del file se necessario

def formatta_lista_somma(lista):
    maiuscole = set()
    risultato = []

    for elemento in lista:
        parte_sinistra, parte_destra = elemento.split(" : ", 1)
        iniziale = parte_sinistra[0]

        if iniziale not in maiuscole:
            maiuscole.add(iniziale)
            risultato.append(parte_sinistra + " : " + parte_destra)

    risultato = sorted(risultato)
    risultato = [elemento.replace("'", '').strip() for elemento in risultato]  # Rimuovi anche i singoli apici

    return risultato

# Funzioni di selezione codice lettera + numero successivo

def associa_valori_alfabeto():
    alfabeto = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    associazioni = {}

    for indice, lettera in enumerate(alfabeto, 1):
        associazioni[lettera] = indice

    return associazioni

def scomponi(parola):
    lettere = ""
    numeri = ""
    for carattere in parola:
        if carattere.isalpha():
            lettere += carattere
        elif carattere.isdigit():
            numeri += carattere
    if numeri != "":
        return lettere, int(numeri)
    else:
        return lettere, 0

def primo_numero_mancante(lista_numeri):
    lista_numeri.sort()
    numero_mancante = 1
    for numero in lista_numeri:
        if numero == numero_mancante:
            numero_mancante += 1
        elif numero > numero_mancante:
            return numero_mancante
    return numero_mancante

def find_next_uppercase_letter(funzioni_importate):
    used_letters = set()
    indici_usati = []
    associazioni = associa_valori_alfabeto()
    valori_numerici = {numero: lettera for lettera, numero in associazioni.items()}

    for funzione in funzioni_importate:
        funzione = funzione.strip("'\"")  # Rimuovi gli apici singoli e doppi
        parts = funzione.split(" : ")
        if len(parts) >= 2:
            letter, number = scomponi(parts[0].strip())
            if letter and letter.isalpha() and letter.isupper():
                # Determino l'indice di ogni codice
                indice = number*26 + associazioni.get(letter)
                indici_usati.append(indice)

    # Determino l'indice mancante
    indice = primo_numero_mancante(indici_usati)
    
    # Qui determino la combinazione lettera + numero dell'indice mancante
    numero = indice // 26
    indice_lettera = indice - numero * 26
    lettera_corrispondente = valori_numerici[indice_lettera]
    if numero == 0:
        return lettera_corrispondente
    else:
        return lettera_corrispondente + str(numero)

def scrivi_lista_linea_successiva(file_name, linea_da_trovare, lista_da_scrivere, nome_lista = 'polinomi'):
    with open(file_name, 'r+') as file:
        contenuto = file.readlines()
        indice_linea = None

        # Trova l'indice della linea da trovare
        for i, linea in enumerate(contenuto):
            if linea.strip().startswith(linea_da_trovare):
                indice_linea = i
                break

        # Se l'indice è stato trovato, scrivi la dichiarazione della lista "funzioni" alla linea successiva
        if indice_linea is not None:
            indice_linea += 1  # Calcola l'indice della linea successiva
            dichiarazione_lista = f'    {nome_lista} = {lista_da_scrivere}\n'
            contenuto.insert(indice_linea, dichiarazione_lista)

            # Scrivi il contenuto aggiornato nel file
            file.seek(0)
            file.writelines(contenuto)
        else:
            print("Linea non trovata nel file.")
            
"""
def lagrange_inter_poly(zeros):
    z = Symbol('z')
    polynomials = []
    for i in range(len(zeros)):
        num = 1
        den = 1
        zi = zeros[i]
        for j in range(len(zeros)):
            zj = zeros[j]
            if zj != zi:
                num *= z - zj
                den *= zi - zj
        poly = num / den
        polynomials.append(poly)
    return polynomials


def integrate_polynomial(polynomial, interval, deg):
    z = Symbol('z')
    a = interval[0]
    b = interval[1]
    polynomial = expand(polynomial)
    integral = 0
    for i in range(deg):
        integral += polynomial.coeff(z, i) / (i+1) * (b ** (i+1) - a ** (i+1))
    return integral
"""
def check(number, epsilon=1e-10):
    """
    Aggiunge un valore piccolissimo a 0 se il numero è 0.
    
    Parametri:
        number (float): Il numero da verificare.
        epsilon (float, opzionale): Il valore piccolissimo da aggiungere a 0.
    
    Ritorna:
        float: Il numero originale se è diverso da 0, altrimenti 0 + epsilon.
    """
    if abs(number) < epsilon:
        return 0.0 + epsilon
    else:
        return number

"""
#######################################################
Funzioni che aiutano nell'inversione delle coordinate a testuggine.
#######################################################
"""

def lista_temporanea(list_, l, j):
    interr = False
    b = list_[j][0]
    a = 0
    h = 0
    copy_list = []
    for k in range(l, -1, -1):
        copy_list.append([list_[k]])
        if list_[k]:
            interr = True
            a = list_[k][0]
            h = (b - a) / len(copy_list)
            break        
    L = len(copy_list)    
    if not interr:
        b = list_[j+1][0]
        h = b - list_[j][0]
        a = b - (j + 1) * h
        k = 0
        L = j    
    temp_list = [[a + h * i] for i in range(L)]
    return temp_list, k
            
def fill_list_with_interpolation(list_):
    filled_list = []
    prev_non_empty_index = None
    j = 0
    h = 0
    num = 0
    temp_list = []
    interruttore = False
    for i in range(len(list_),0,-1):
        if not list_[i - 1]:
            lista_temp, k = lista_temporanea(list_, i - 1, i)
            list_[k:i] = lista_temp[:]
            i = k
    return list_

def numeri_piu_vicini_a_zero(f, lin):
    ys = f(lin)
    ypos = 10
    yneg = -10
    numero_piu_vicino_negativo = None
    posizPos = 0
    posizNeg = 0
    for i in range(len(ys)):
        numero = ys[i]
        if numero > 0:
            if ypos is None or numero < ypos:
                ypos = numero
                posizPos = i
        elif numero < 0:
            if yneg is None or numero > yneg:
                yneg = numero
                posizNeg = i
    xpos = lin[posizPos]
    xneg = lin[posizNeg]
    return ypos, xpos, yneg, xneg
