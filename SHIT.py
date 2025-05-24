

from abc import ABC, abstractmethod
import csv
import os
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

# ================== CLASSES BASE ==================
class Usuari:
    def __init__(self, user_id: int, location: str, age: float):
        self._id = user_id
        self._location = location
        self._age = age

    def get_id(self) -> int:
        return self._id

class Item(ABC):
    def __init__(self, item_id: Union[int, str], titol: str, categoria: str):
        self._id = item_id
        self._titol = titol
        self._categoria = categoria

    def get_id(self):
        return self._id

    def get_titol(self) -> str:
        return self._titol

    def get_categoria(self) -> str:
        return self._categoria

# ============== CLASSES ESPECÍFIQUES ==============
class Llibre(Item):
    def __init__(self, item_id: str, titol: str, any_publicacio: int, autor: str, editorial: str = "Desconeguda"):
        super().__init__(item_id, titol, "Llibre")
        self._any = any_publicacio
        self._autor = autor
        self._editorial = editorial

    def get_info(self) -> str:
        return f"Autor: {self._autor}, Any: {self._any}, Editorial: {self._editorial}"

class Peli(Item):
    def __init__(self, item_id: int, titol: str, genere: str, imdb_id: int = 0, tmdb_id: int = 0):
        super().__init__(item_id, titol, "Peli")
        self._genere = genere

    def get_info(self) -> str:
        return f"Gènere: {self._genere}"

# ============== CLASSE ABSTRACTA DADES ==============
class Dades(ABC):
    def __init__(self):
        self._ratings_matrix: np.ndarray = np.array([])
        self._users: Dict[int, Usuari] = {}
        self._items: Dict[Union[int, str], Item] = {}
        self._user_id_to_idx: Dict[int, int] = {}
        self._item_id_to_idx: Dict[Union[int, str], int] = {}

    def get_usuari(self, user_id: int) -> Optional[Usuari]:
        return self._users.get(user_id)

    def get_item(self, item_id: Union[int, str]) -> Optional[Item]:
        return self._items.get(item_id)

    def get_rating_matrix(self) -> np.ndarray:
        return self._ratings_matrix

    @abstractmethod
    def carregar_usuaris(self, path: str):
        pass

    @abstractmethod
    def carregar_items(self, path: str):
        pass

    @abstractmethod
    def carregar_valoracions(self, path: str):
        pass

# ============== IMPLEMENTACIÓ PELÍCULES ==============
class DadesPelis(Dades):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def _carregar_csv(self, fitxer: str) -> List[List[str]]:
        try:
            with open(fitxer, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                return [line for line in reader if line]
        except Exception as e:
            print(f"Error llegint {fitxer}: {str(e)}")
            return []

    def carregar_usuaris(self, path: str):
        data = self._carregar_csv(path)
        self._users = {}
        for line in data:
            if len(line) >= 3:
                try:
                    user_id = int(line[0])
                    self._users[user_id] = Usuari(user_id, "", 0.0)
                except ValueError:
                    continue
        self._user_id_to_idx = {uid: idx for idx, uid in enumerate(sorted(self._users))}

    def carregar_items(self, path: str):
        data = self._carregar_csv(path)
        self._items = {}
        self._item_id_to_idx = {}
        for idx, line in enumerate(data):
            if len(line) >= 3:
                try:
                    movie_id = int(line[0])
                    title = line[1]
                    genres = line[2]
                    self._items[movie_id] = Peli(movie_id, title, genres)
                    self._item_id_to_idx[movie_id] = idx
                except ValueError:
                    continue

    def carregar_valoracions(self, path: str):
        num_users = len(self._user_id_to_idx)
        num_items = len(self._item_id_to_idx)
        self._ratings_matrix = np.zeros((num_users, num_items), dtype=np.float32)

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                if len(line) >= 3:
                    try:
                        user_id = int(line[0])
                        movie_id = int(line[1])
                        rating = float(line[2])
                        if rating <= 0 or rating > 5:
                            continue
                        if user_id in self._user_id_to_idx and movie_id in self._item_id_to_idx:
                            u_idx = self._user_id_to_idx[user_id]
                            i_idx = self._item_id_to_idx[movie_id]
                            self._ratings_matrix[u_idx, i_idx] = rating
                    except ValueError:
                        continue


# ============== IMPLEMENTACIÓ LLIBRES (CORREGIDA) ==============
class DadesLlibres(Dades):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self.MAX_BOOKS = 10000  # Només llegir 10k línies

    def _carregar_csv(self, fitxer: str) -> List[List[str]]:
        try:
            with open(fitxer, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Saltar capçalera
                return [line for idx, line in enumerate(reader) if idx < self.MAX_BOOKS and line]  # Llegir fins a 10k línies
        except Exception as e:
            print(f"Error llegint {fitxer}: {str(e)}")
            return []
        
    def carregar_usuaris(self, path: str):
        data = self._carregar_csv(path)
        self._users = {}
        for line in data:
            if len(line) >= 3:
                try:
                    user_id = int(line[0])
                    location = line[1]
                    age = float(line[2]) if line[2].strip() else 0.0
                    self._users[user_id] = Usuari(user_id, location, age)
                except ValueError:
                    continue
        self._user_id_to_idx = {uid: idx for idx, uid in enumerate(sorted(self._users))}

    def carregar_items(self, path: str):
        data = self._carregar_csv(path)
        self._items = {}
        self._item_id_to_idx = {}
        
        for idx, line in enumerate(data):
            if len(line) >= 4:
                isbn = line[0]
                titol = line[1]
                autor = line[2]
                any_publicacio = int(line[3]) if line[3].strip().isdigit() else 0
                editorial = line[4] if len(line) > 4 else "Desconeguda"
                
                self._items[isbn] = Llibre(isbn, titol, any_publicacio, autor, editorial)
                self._item_id_to_idx[isbn] = idx  # Índex segons l'ordre de lectura (sense ordenar!)

    def carregar_valoracions(self, path: str):
        # 1. Crear matriu amb mides correctes
        num_users = len(self._user_id_to_idx)
        num_items = len(self._item_id_to_idx)
        self._ratings_matrix = np.zeros((num_users, num_items), dtype=np.float32)
        
        # 2. Carregar només valoracions d'usuaris i ítems dins dels 10k
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                if len(line) >= 3:
                    try:
                        user_id = int(line[0])
                        isbn = line[1]
                        rating = float(line[2])
                        
                        # Verificar que l'usuari i l'ítem existeixen als diccionaris
                        if user_id in self._user_id_to_idx and isbn in self._item_id_to_idx:
                            u_idx = self._user_id_to_idx[user_id]
                            i_idx = self._item_id_to_idx[isbn]
                            
                            if 1 <= rating <= 10:  # Acceptar valors entre 1 i 10
                                self._ratings_matrix[u_idx, i_idx] = rating
                    except ValueError:
                        continue
        
        print(f"[DEBUG] Valoracions no zero: {np.count_nonzero(self._ratings_matrix)}")


# ============== RECOMANADOR COL·LABORATIU ==============
class Recomanador(ABC):
    def __init__(self, dades: Dades):
        self._dades = dades

    @abstractmethod
    def recomana(self, user_id: int, n: int) -> List[Tuple[Item, float]]:
        pass


class RecomanadorSimple(Recomanador):
    def __init__(self, dades: Dades, min_vots: int = 3):
        super().__init__(dades)
        self._min_vots = min_vots

    def recomana(self, user_id: int, n: int = 10) -> List[Tuple[Item, float]]:
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        matriu = self._dades.get_rating_matrix()
        valoracions_usuari = matriu[user_idx]
        #print(valoracions_usuari)
        items_valorats = set(np.where(valoracions_usuari > 0)[0])
        #print(items_valorats)
        

        total_ratings = matriu[matriu > 0]
        avg_global = np.mean(total_ratings) if total_ratings.size > 0 else 0.0

        prediccions = []
        for item_idx in range(matriu.shape[1]):
            if item_idx in items_valorats:
                continue

            ratings_item = matriu[:, item_idx]
            valid_ratings = ratings_item[ratings_item > 0]
            num_vots = len(valid_ratings)

            if num_vots < self._min_vots:
                continue

            avg_item = np.mean(valid_ratings)
            score = (num_vots / (num_vots + self._min_vots)) * avg_item + \
                    (self._min_vots / (num_vots + self._min_vots)) * avg_global

            item_id = list(self._dades._item_id_to_idx.keys())[item_idx]
            item = self._dades.get_item(item_id)
            if item:
                prediccions.append((item, score))

        prediccions.sort(key=lambda x: x[1], reverse=True)
        return prediccions[:n]

class RecomanadorCol·laboratiu(Recomanador):
    def __init__(self, dades: Dades, k: int = 50):  # Augmentar k significativament
        super().__init__(dades)
        self._k = k

    def recomana(self, user_id: int, n: int = 10) -> List[Tuple[Item, float]]:
        matriu = self._dades.get_rating_matrix()
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            print("⚠️ Usuari no trobat")
            return []

        user_ratings = matriu[user_idx, :]
        valoracions_valides = user_ratings[user_ratings > 0]
        if len(valoracions_valides) == 0:
            print("⚠️ L'usuari no té valoracions")
            return []
            
        mu_u = np.mean(valoracions_valides)

        similituds = []
        for alt_idx in range(matriu.shape[0]):
            if alt_idx == user_idx:
                continue
            
            # Trobar ítems comuns AMB VALORACIONS (>0)
            mascara_comuns = (user_ratings > 0) & (matriu[alt_idx] > 0)
            items_comuns = np.where(mascara_comuns)[0]
            
            if len(items_comuns) < 1:  # Permetre mínim 1 ítem comú
                continue
            
            u = user_ratings[items_comuns]
            v = matriu[alt_idx][items_comuns]
            
            # Càlcul robust de similitud del cosinus
            producte_punt = np.dot(u, v)
            norma_u = np.linalg.norm(u)
            norma_v = np.linalg.norm(v)
            
            if norma_u == 0 or norma_v == 0:
                sim = 0.0
            else:
                sim = producte_punt / (norma_u * norma_v)
            
            similituds.append((alt_idx, sim))

        # Seleccionar TOP k veïns (fins i tot amb similitud baixa)
        similituds.sort(key=lambda x: x[1], reverse=True)
        top_k = similituds[:self._k]

        # Si no hi ha veïns, retornar buit
        if not top_k:
            print("⚠️ No s'han trobat usuaris similars")
            return []

        scores = np.zeros(matriu.shape[1])
        weights = np.zeros(matriu.shape[1])

        for alt_idx, sim in top_k:
            alt_ratings = matriu[alt_idx]
            mu_v = np.mean(alt_ratings[alt_ratings > 0]) if np.any(alt_ratings > 0) else 0.0
            
            for i in range(matriu.shape[1]):
                if user_ratings[i] == 0 and alt_ratings[i] > 0:
                    scores[i] += sim * (alt_ratings[i] - mu_v)
                    weights[i] += abs(sim)

        # Generar recomanacions només per ítems no valorats i amb pes > 0
        recomanacions = []
        for i in range(matriu.shape[1]):
            if weights[i] > 0 and user_ratings[i] == 0:
                pred = mu_u + (scores[i] / weights[i])
                item_id = list(self._dades._item_id_to_idx.keys())[i]
                item = self._dades.get_item(item_id)
                if item and pred > 0:  # Filtrar prediccions negatives
                    recomanacions.append((item, pred))
        
        if not recomanacions:
            print("⚠️ No hi ha suficients dades per generar recomanacions")
            return []

        recomanacions.sort(key=lambda x: x[1], reverse=True)
        return recomanacions[:n]
    

def main():
    print("=== SISTEMA DE RECOMANACIÓ ===")
    tipus = input("Selecciona el tipus de dades (llibres/pelis): ").lower()
    
    if tipus == "llibres":
        dades = DadesLlibres("carpeta_books/")
        dades.carregar_usuaris("carpeta_books/Users.csv")
        dades.carregar_items("carpeta_books/Books.csv")
        dades.carregar_valoracions("carpeta_books/Ratings.csv")
    elif tipus == "pelis":
        dades = DadesPelis("carpeta_movies/")
        dades.carregar_usuaris("carpeta_movies/ratings.csv")
        dades.carregar_items("carpeta_movies/movies.csv")
        dades.carregar_valoracions("carpeta_movies/ratings.csv")
    else:
        print("Tipus de dades no vàlid")
        return

    tipus_rec = input("Selecciona el tipus de recomanador (simple/col·laboratiu): ").lower()
    if tipus_rec == "simple":
        recomanador = RecomanadorSimple(dades)
    elif tipus_rec == "col·laboratiu":
        recomanador = RecomanadorCol·laboratiu(dades)
    else:
        print("Tipus de recomanador no vàlid")
        return

    while True:
        user_input = input("\nIntrodueix ID d'usuari (ENTER per sortir): ").strip()
        if not user_input:
            break
        try:
            user_id = int(user_input)
            recomanacions = recomanador.recomana(user_id, 5)
            
            if not recomanacions:
                print("No hi ha recomanacions disponibles")
            else:
                print(f"\nTop 5 recomanacions per {user_id}:")
                for idx, (item, puntuacio) in enumerate(recomanacions, 1):
                    print(f"{idx}. {item.get_titol()} ({item.get_categoria()})")
                    if isinstance(item, Llibre):
                        print(f"   Autor: {item.get_info()}")
                    elif isinstance(item, Peli):
                        print(f"   Gènere: {item.get_info()}")
                    print(f"   Puntuació estimada: {puntuacio:.2f}\n")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()



"""

dades = DadesLlibres("carpeta_books/")
dades.carregar_usuaris("carpeta_books/Users.csv")
dades.carregar_items("carpeta_books/Books.csv")
dades.carregar_valoracions("carpeta_books/Ratings.csv")

def mostrar_usuaris_actius(dades: Dades):
    matriu = dades.get_rating_matrix()
    usuaris_actius = []
    
    for user_id in dades._users:
        user_idx = dades._user_id_to_idx.get(user_id)
        if user_idx is not None:
            # Comprovar si té alguna valoració > 0
            if np.any(matriu[user_idx] > 0):
                usuaris_actius.append(user_id)
    
    print(f"Usuaris amb valoracions: {len(usuaris_actius)}/{len(dades._users)}")
    print("Llista d'IDs actius:", usuaris_actius)

mostrar_usuaris_actius(dades)

"""