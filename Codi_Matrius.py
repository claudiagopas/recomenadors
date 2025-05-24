import csv
import os
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod

# === CLASSES BASE ===
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

# === CLASSES ESPECÍFIQUES ===
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
        self._imdb_id = imdb_id
        self._tmdb_id = tmdb_id

    def get_info(self) -> str:
        return f"Gènere: {self._genere}, IMDb: {self._imdb_id}"

# === CLASSE ABSTRACTA DADES ===
class Dades(ABC):
    def __init__(self):
        self._ratings_matrix = sp.csc_matrix((0, 0), dtype=np.float32)
        self._users: Dict[int, Usuari] = {}  # Optimització amb diccionari
        self._items: Dict[Union[int, str], Item] = {} 
        self._user_id_to_idx: Dict[int, int] = {}
        self._item_id_to_idx: Dict[Union[int, str], int] = {}

    def get_usuari(self, user_id: int) -> Optional[Usuari]:
        return self._users.get(user_id)

    def get_item(self, item_id: Union[int, str]) -> Optional[Item]:
        return self._items.get(item_id)
    
    def get_rating_matrix(self) -> sp.csc_matrix:
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

# === IMPLEMENTACIONS CONCRETES ===
class DadesLlibres(Dades):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def _carregar_csv(self, fitxer: str) -> List[List[str]]:
        try:
            with open(fitxer, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Saltar capçalera
                return [line for line in reader if line]
        except Exception as e:
            print(f"Error llegint {fitxer}: {str(e)}")
            return []

    def carregar_usuaris(self, path: str):
        data = self._carregar_csv(path)
        self._users = {}
        self._user_id_to_idx = {}
        for idx, line in enumerate(data):
            if len(line) < 3:  # Validar longitud mínima
                print(f"Línia invàlida a Users.csv: {line}")
                continue
            user_id_str, location, age_str = line[0], line[1], line[2]
            try:
                user_id = int(user_id_str)
                age = float(age_str) if age_str.strip() else 0.0
            except ValueError:
                print(f"ID o edat invàlids a Users.csv: {line}")
                continue
            self._users[user_id] = Usuari(user_id, location, age)
            self._user_id_to_idx[user_id] = idx

    def carregar_items(self, path: str):
        data = self._carregar_csv(path)
        self._items = {}
        self._item_id_to_idx = {}
        for idx, line in enumerate(data):
            if len(line) < 4:  # Validar ISBN, títol, autor, any
                print(f"Línia invàlida a Books.csv: {line}")
                continue
            isbn, title, author, year_str = line[0], line[1], line[2], line[3]
            publisher = line[4] if len(line) > 4 else "Desconeguda"
            try:
                year = int(year_str) if year_str.strip().isdigit() else 0  # Correcció any
            except ValueError:
                year = 0
            self._items[isbn] = Llibre(isbn, title, year, author, publisher)
            self._item_id_to_idx[isbn] = idx

    def carregar_valoracions(self, path: str):
        data = self._carregar_csv(path)
        rows, cols, data_vals = [], [], []
        for line in data:
            if len(line) < 3:
                print(f"Línia invàlida a Ratings.csv: {line}")
                continue
            user_id_str, isbn, rating_str = line[0], line[1], line[2]
            try:
                user_id = int(user_id_str)
                rating = float(rating_str)
                if rating <= 0 or rating > 10:  # Validar rang (ex: 1-10)
                    continue
            except ValueError:
                print(f"Valoració invàlida a Ratings.csv: {line}")
                continue
            user = self._users.get(user_id)
            item = self._items.get(isbn)
            if user and item:
                user_idx = self._user_id_to_idx[user_id]
                item_idx = self._item_id_to_idx[isbn]
                rows.append(user_idx)
                cols.append(item_idx)
                data_vals.append(rating)
        if data_vals:
            num_users = len(self._users)
            num_items = len(self._items)
            self._ratings_matrix = sp.coo_matrix(
                (data_vals, (rows, cols)), 
                shape=(num_users, num_items)
            ).tocsc()
        else:
            self._ratings_matrix = sp.csc_matrix((len(self._users), len(self._items)), dtype=np.float32)

class DadesPelis(Dades):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._metadata = {'links': [], 'tags': []}

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
        # Processar tots els user_ids únics de ratings i tags
        base_dir = os.path.dirname(path)
        ratings_data = self._carregar_csv(path)
        tags_path = os.path.join(base_dir, 'tags.csv')
        tags_data = self._carregar_csv(tags_path)
        user_ids = set()
        for line in ratings_data:
            if len(line) >= 1:
                try:
                    user_ids.add(int(line[0]))
                except ValueError:
                    pass
        for line in tags_data:
            if len(line) >= 1:
                try:
                    user_ids.add(int(line[0]))
                except ValueError:
                    pass
        self._users = {}
        self._user_id_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}
        for user_id in user_ids:
            self._users[user_id] = Usuari(user_id, "", 0.0)

    def carregar_items(self, path: str):
        data = self._carregar_csv(path)
        self._items = {}
        self._item_id_to_idx = {}
        movie_id_to_idx = {}  # Optimització per a links
        for idx, line in enumerate(data):
            if len(line) < 3:
                print(f"Línia invàlida a movies.csv: {line}")
                continue
            movie_id_str, title, genres = line[0], line[1], line[2]
            try:
                movie_id = int(movie_id_str)
            except ValueError:
                print(f"ID invàlid a movies.csv: {line}")
                continue
            self._items[movie_id] = Peli(movie_id, title, genres)
            movie_id_to_idx[movie_id] = idx
            self._item_id_to_idx[movie_id] = idx

        # Processar links amb diccionari
        for link in self._metadata['links']:
            if len(link) < 3:
                continue
            movie_id_str, imdb_id_str, tmdb_id_str = link[0], link[1], link[2]
            try:
                movie_id = int(movie_id_str)
                imdb_id = int(imdb_id_str) if imdb_id_str else 0
                tmdb_id = int(tmdb_id_str) if tmdb_id_str else 0
            except ValueError:
                continue
            if movie_id in movie_id_to_idx:
                idx = movie_id_to_idx[movie_id]
                self._items[movie_id]._imdb_id = imdb_id
                self._items[movie_id]._tmdb_id = tmdb_id

    def carregar_valoracions(self, path: str):
        data = self._carregar_csv(path)
        rows, cols, data_vals = [], [], []
        for line in data:
            if len(line) < 4:
                print(f"Línia invàlida a ratings.csv: {line}")
                continue
            user_id_str, movie_id_str, rating_str, _ = line[0], line[1], line[2], line[3]
            try:
                user_id = int(user_id_str)
                movie_id = int(movie_id_str)
                rating = float(rating_str)
                if rating < 0 or rating > 5:  # Validar rang 0-5
                    continue
            except ValueError:
                print(f"Valoració invàlida a ratings.csv: {line}")
                continue
            user = self._users.get(user_id)
            item = self._items.get(movie_id)
            if user and item:
                user_idx = self._user_id_to_idx[user_id]
                item_idx = self._item_id_to_idx[movie_id]
                rows.append(user_idx)
                cols.append(item_idx)
                data_vals.append(rating)
        if data_vals:
            num_users = len(self._users)
            num_items = len(self._items)
            self._ratings_matrix = sp.coo_matrix(
                (data_vals, (rows, cols)), 
                shape=(num_users, num_items)
            ).tocsc()
        else:
            self._ratings_matrix = sp.csc_matrix((len(self._users), len(self._items)), dtype=np.float32)

    def carregar_links(self, path: str):
        self._metadata['links'] = self._carregar_csv(path)

    def carregar_tags(self, path: str):
        self._metadata['tags'] = self._carregar_csv(path)

# === RESTA DEL CODI (Recomanador, main) ES MANTÉ IGUAL ===

# === SISTEMA DE RECOMANACIÓ ===
class Recomanador(ABC):
    def __init__(self, dades: Dades):
        self._dades = dades

    @abstractmethod
    def recomana(self, user_id: int, n: int) -> List[Tuple[Item, float]]:
        pass

# === SISTEMA DE RECOMANACIÓ ===
class RecomanadorSimple(Recomanador):
    def __init__(self, dades: Dades, min_vots: int = 3):
        super().__init__(dades)
        self._min_vots = min_vots

    def recomana(self, user_id: int, n: int = 5) -> List[Tuple[Item, float]]:
        usuari = self._dades.get_usuari(user_id)
        if not usuari:
            raise ValueError(f"Usuari {user_id} no trobat")
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []
        
        user_ratings = self._dades._ratings_matrix.getrow(user_idx)
        items_valorats = user_ratings.indices  # Índexs dels items valorats

        # Càlcul de la mitjana global (excloent 0s)
        total_ratings = self._dades._ratings_matrix.data
        avg_global = np.mean(total_ratings) if total_ratings.size > 0 else 0.0

        puntuacions = []
        # Iterar sobre els OBJECTES Item (valors del diccionari)
        for item in self._dades._items.values():  
            item_idx = self._dades._item_id_to_idx[item.get_id()]
            
            if item_idx in items_valorats:
                continue  # Saltar items ja valorats
            
            # Obtenir valoracions de l'ítem (només dades != 0)
            item_ratings = self._dades._ratings_matrix.getcol(item_idx)
            valid_ratings = item_ratings.data
            num_vots = valid_ratings.size
            
            if num_vots < self._min_vots:
                continue  # Filtrar per mínim de vots
            
            avg_item = np.mean(valid_ratings) if num_vots > 0 else 0.0
            
            # Fórmula de ponderació
            score = (num_vots / (num_vots + self._min_vots)) * avg_item
            score += (self._min_vots / (num_vots + self._min_vots)) * avg_global
            puntuacions.append((item, score))  # Guardar l'objecte Item
            
        puntuacions.sort(key=lambda x: x[1], reverse=True)
        return puntuacions[:n]

# === MAIN ===
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
        dades.carregar_links("carpeta_movies/links.csv")
        dades.carregar_tags("carpeta_movies/tags.csv")
    else:
        print("Tipus no vàlid")
        return

    recomanador = RecomanadorSimple(dades, min_vots=3)

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
                    # Assegurar-se que 'item' és un objecte Item
                    print(f"{idx}. {item.get_titol()} ({item.get_categoria()})") 
                    if isinstance(item, Llibre):  
                        print(f"   {item.get_info()}")
                    elif isinstance(item, Peli): 
                        print(f"   {item.get_info()}")
                    print(f"   Puntuació estimada: {puntuacio:.2f}\n")
        except ValueError:
            print("ID ha de ser un número enter")
        except Exception as e:
            print(f"Error: {str(e)}")
    
if __name__ == "__main__":
    main()
