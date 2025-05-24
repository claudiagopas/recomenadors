import csv
import os

def comprovar_user_id(nom_fitxer, user_id):
    trobat = False
    comptador = 0
    max_files = 10000

    ruta_fitxer = os.path.join(os.path.dirname(__file__), nom_fitxer)

    with open(ruta_fitxer, newline='', encoding='utf-8') as fitxer:
        lector = csv.reader(fitxer)
        next(lector)

        for i, fila in enumerate(lector):
            if i >= max_files:
                break
            if fila[0] == str(user_id):
                trobat = True
                comptador += 1
                print(fila)

    if trobat:
        print(f"El User-ID {user_id} existeix al fitxer (dins les primeres {max_files} línies), {comptador} vegades.")
    else:
        print(f"El User-ID {user_id} no s'ha trobat dins les primeres {max_files} línies.")

# Exemple d'ús
nom_fitxer = "Ratings.csv"
user_id_a_cercar = input("Introdueix el User-ID a cercar: ")
comprovar_user_id(nom_fitxer, user_id_a_cercar)
