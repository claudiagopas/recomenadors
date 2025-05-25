# CODI COMPLET FASE FINAL DEL PROJECTE

# IMPORTACIONS I CONFIGURACIÓ
import os
import sys
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod

now = datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"log_{now}.txt"),
        logging.StreamHandler()
    ]
)

# [INCLÒS: totes les classes Usuari, Item, Llibre, Peli, Dades, DadesLlibres, DadesPelis]
# [INCLÒS: RecomanadorSimple, RecomanadorCol·laboratiu, RecomenadorContingut (placeholder)]
# [INCLÒS: Classe Avaluador amb MAE i RMSE]

# === FUNCIONS D'AJUT ===
def carregar_amb_pickle(dataset: str, method: str):
    nom_fitxer = f"recommender_{dataset}_{method}.dat"
    if os.path.exists(nom_fitxer):
        logging.info(f"Carregant recomanador des de {nom_fitxer}")
        with open(nom_fitxer, "rb") as f:
            return pickle.load(f)
    return None

def desa_amb_pickle(obj, dataset: str, method: str):
    nom_fitxer = f"recommender_{dataset}_{method}.dat"
    logging.info(f"Desant recomanador a {nom_fitxer}")
    with open(nom_fitxer, "wb") as f:
        pickle.dump(obj, f)


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

    def _prediccio_item(self, item_idx: int, avg_global: float) -> Optional[float]:
        """
        Calcula el score (predicció) per un ítem donat.
        """
        matriu = self._dades.get_rating_matrix()
        ratings_item = matriu[:, item_idx]
        valid_ratings = ratings_item[ratings_item > 0]
        num_vots = len(valid_ratings)

        if num_vots < self._min_vots:
            return None

        avg_item = np.mean(valid_ratings)
        score = (num_vots / (num_vots + self._min_vots)) * avg_item + \
                (self._min_vots / (num_vots + self._min_vots)) * avg_global
        return score

    def recomana(self, user_id: int, n: int = 10) -> List[Tuple[Item, float]]:
        matriu = self._dades.get_rating_matrix()
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        valoracions_usuari = matriu[user_idx]
        items_valorats = set(np.where(valoracions_usuari > 0)[0])
        avg_global = np.mean(matriu[matriu > 0]) if np.count_nonzero(matriu) > 0 else 0.0
        if avg_global == 0.0:
            logging.warning("No hi ha valoracions globals per calcular la mitjana.")
            return []

        prediccions = []
        for item_idx in range(matriu.shape[1]):
            if item_idx in items_valorats:
                continue

            score = self._prediccio_item(item_idx, avg_global)
            if score is not None:
                item_id = list(self._dades._item_id_to_idx.keys())[item_idx]
                item = self._dades.get_item(item_id)
                if item:
                    prediccions.append((item, score))

        prediccions.sort(key=lambda x: x[1], reverse=True)
        return prediccions[:n]

    def prediu(self, user_id: int, item_ids: List[Union[int, str]]) -> List[Tuple[Item, float]]:
        matriu = self._dades.get_rating_matrix()
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        avg_global = np.mean(matriu[matriu > 0]) if np.count_nonzero(matriu) > 0 else 0.0
        prediccions = []

        for item_id in item_ids:
            item_idx = self._dades._item_id_to_idx.get(item_id)
            if item_idx is None:
                continue

            score = self._prediccio_item(item_idx, avg_global)
            if score is not None:
                item = self._dades.get_item(item_id)
                if item:
                    prediccions.append((item, score))

        return prediccions
class RecomanadorCol·laboratiu(Recomanador):
    def __init__(self, dades: Dades, k: int = 10):
        super().__init__(dades)
        self._k = k

    def _similituds_usuari(self, user_idx: int, matriu: np.ndarray) -> List[Tuple[int, float]]:
        user_ratings = matriu[user_idx, :]
        similituds = []

        for alt_idx in range(matriu.shape[0]):
            if alt_idx == user_idx:
                continue

            comuns = (user_ratings > 0) & (matriu[alt_idx] > 0)
            if np.count_nonzero(comuns) < 1:
                continue

            u = user_ratings[comuns]
            v = matriu[alt_idx, comuns]

            prod = np.dot(u, v)
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            sim = prod / (norm_u * norm_v) if norm_u != 0 and norm_v != 0 else 0.0
            similituds.append((alt_idx, sim))

        similituds.sort(key=lambda x: x[1], reverse=True)
        return similituds[:self._k]

    def _prediccio_item(self, user_idx: int, item_idx: int, similituds: List[Tuple[int, float]], mu_u: float, matriu: np.ndarray) -> Optional[float]:
        numerador = 0.0
        denominador = 0.0

        for alt_idx, sim in similituds:
            rating = matriu[alt_idx, item_idx]
            if rating > 0:
                mu_v = np.mean(matriu[alt_idx][matriu[alt_idx] > 0]) if np.any(matriu[alt_idx] > 0) else 0.0
                numerador += sim * (rating - mu_v)
                denominador += abs(sim)

        if denominador == 0:
            return None

        return mu_u + (numerador / denominador)

    def recomana(self, user_id: int, n: int = 10) -> List[Tuple[Item, float]]:
        matriu = self._dades.get_rating_matrix()
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        user_ratings = matriu[user_idx]
        if np.any(user_ratings >0):
            mu_u = np.mean(user_ratings[user_ratings > 0]) 
        else:
            mu_u = 0.0

        if mu_u == 0.0:
            logging.warning("Usuari sense valoracions per calcular la mitjana.")
            return []    

        similituds = self._similituds_usuari(user_idx, matriu)

        prediccions = []
        for item_idx in range(matriu.shape[1]):
            if user_ratings[item_idx] > 0:
                continue  # ja valorat

            pred = self._prediccio_item(user_idx, item_idx, similituds, mu_u, matriu)
            if pred and pred > 0:
                item_id = list(self._dades._item_id_to_idx.keys())[item_idx]
                item = self._dades.get_item(item_id)
                if item:
                    prediccions.append((item, pred))

        prediccions.sort(key=lambda x: x[1], reverse=True)
        return prediccions[:n]

    def prediu(self, user_id: int, item_ids: List[Union[int, str]]) -> List[Tuple[Item, float]]:
        matriu = self._dades.get_rating_matrix()
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        user_ratings = matriu[user_idx]
        mu_u = np.mean(user_ratings[user_ratings > 0]) if np.any(user_ratings > 0) else 0.0
        similituds = self._similituds_usuari(user_idx, matriu)

        prediccions = []
        for item_id in item_ids:
            item_idx = self._dades._item_id_to_idx.get(item_id)
            if item_idx is None:
                continue

            pred = self._prediccio_item(user_idx, item_idx, similituds, mu_u, matriu)
            if pred and pred > 0:
                item = self._dades.get_item(item_id)
                if item:
                    prediccions.append((item, pred))

        return prediccions

class RecomanadorContingut(Recomanador):
    def __init__(self, dades: Dades):
        super().__init__(dades)
        self._tfidf_matrix = None
        self._vocabulari = None
        self._item_ids_ordenats = list(sorted(self._dades._item_id_to_idx, key=lambda k: self._dades._item_id_to_idx[k]))
        self.valoracio_maxima = 5.0 if isinstance(dades, DadesPelis) else 10.0
        self._prepara_tfidf()

    def _prepara_tfidf(self):
        item_features = []
        for item_id in self._item_ids_ordenats:
            item = self._dades.get_item(item_id)
            if isinstance(item, Peli):
                features = item._genere.replace('|', ' ')
            elif isinstance(item, Llibre):
                features = f"{item._autor} {item._editorial} {item._any}"
            else:
                features = ""
            item_features.append(features)

        tfidf = TfidfVectorizer(stop_words='english')
        self._tfidf_matrix = tfidf.fit_transform(item_features).toarray()
        self._vocabulari = tfidf.get_feature_names_out()
        logging.info(f"TF-IDF creat: {self._tfidf_matrix.shape[0]} ítems, {self._tfidf_matrix.shape[1]} característiques")

    def _calcular_perfil_usuari(self, user_idx: int) -> Optional[np.ndarray]:
        matriu = self._dades.get_rating_matrix()
        valoracions = matriu[user_idx]
        items_valorats = np.where(valoracions > 0)[0]

        if len(items_valorats) == 0:
            return None

        rated_ratings = valoracions[items_valorats]
        rated_tfidf = self._tfidf_matrix[items_valorats]

        numerador = np.sum(rated_ratings[:, np.newaxis] * rated_tfidf, axis=0)
        denominador = np.sum(rated_ratings)

        return numerador / denominador if denominador > 0 else None

    def _calcula_similituds(self, perfil: np.ndarray) -> np.ndarray:
        return cosine_similarity(perfil.reshape(1, -1), self._tfidf_matrix)[0]

    def recomana(self, user_id: int, n: int = 5) -> List[Tuple[Item, float]]:
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        perfil = self._calcular_perfil_usuari(user_idx)
        if perfil is None:
            return []

        similituds = self._calcula_similituds(perfil)
        valoracions = self._dades.get_rating_matrix()[user_idx]
        unrated = np.where(valoracions == 0)[0]

        puntuacions = similituds * self.valoracio_maxima
        top_indices = unrated[np.argsort(puntuacions[unrated])[::-1][:n]]

        recomanacions = []
        for idx in top_indices:
            item_id = self._item_ids_ordenats[idx]
            item = self._dades.get_item(item_id)
            if item:
                recomanacions.append((item, puntuacions[idx]))

        return recomanacions

    def prediu(self, user_id: int, item_ids: List[Union[int, str]]) -> List[Tuple[Item, float]]:
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        perfil = self._calcular_perfil_usuari(user_idx)
        if perfil is None:
            return []

        similituds = self._calcula_similituds(perfil)
        prediccions = []

        for item_id in item_ids:
            item_idx = self._dades._item_id_to_idx.get(item_id)
            if item_idx is not None:
                score = similituds[item_idx] * self.valoracio_maxima
                item = self._dades.get_item(item_id)
                if item:
                    prediccions.append((item, score))

        return prediccions


class Avaluador:
    """
    Classe per avaluar la qualitat de les prediccions d'un recomanador.
    """

    def __init__(self, recomanador: Recomanador, dades: Dades):
        self._recomanador = recomanador
        self._dades = dades

    def avalua(self, user_id: int) -> Tuple[List[Tuple[Item, float]], List[Tuple[Item, float]], float, float]:
        """
        Avalua les prediccions fetes pel recomanador per ítems ja valorats per l'usuari.

        Returns
        -------
        (prediccions, valoracions_reals, mae, rmse)
        """
        matriu = self._dades.get_rating_matrix()
        user_idx = self._dades._user_id_to_idx.get(user_id)
        if user_idx is None:
            print("Usuari no trobat.")
            return [], [], 0.0, 0.0

        user_ratings = matriu[user_idx]
        valorats_idx = np.where(user_ratings > 0)[0]
        if len(valorats_idx) == 0:
            print("Usuari sense valoracions.")
            return [], [], 0.0, 0.0

        item_ids = [list(self._dades._item_id_to_idx.keys())[i] for i in valorats_idx]
        prediccions = self._recomanador.prediu(user_id, item_ids)

        y_true = []
        y_pred = []
        valoracions_reals = []

        for item, score in prediccions:
            item_idx = self._dades._item_id_to_idx[item.get_id()]
            real = matriu[user_idx, item_idx]
            y_true.append(real)
            y_pred.append(score)
            valoracions_reals.append((item, real))

        if not y_true:
            print("No hi ha ítems en comú per comparar.")
            return prediccions, valoracions_reals, 0.0, 0.0

        mae = np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
        rmse = np.sqrt(np.mean((np.array(y_pred) - np.array(y_true)) ** 2))

        return prediccions, valoracions_reals, mae, rmse
    
# === MAIN ===
def main():
    if len(sys.argv) != 3:
        print("Ús: python main.py <dataset> <method>")
        return

    dataset, method = sys.argv[1].lower(), sys.argv[2].lower()
    recomanador = carregar_amb_pickle(dataset, method)

    if not recomanador:
        if dataset == "pelis":
            dades = DadesPelis("dataset/MovieLens100k/")
            dades.carregar_usuaris("dataset/MovieLens100k/ratings.csv")
            dades.carregar_items("dataset/MovieLens100k/movies.csv")
            dades.carregar_valoracions("dataset/MovieLens100k/ratings.csv")
        elif dataset == "llibres":
            dades = DadesLlibres("dataset/Books/")
            dades.carregar_usuaris("dataset/Books/Users.csv")
            dades.carregar_items("dataset/Books/Books.csv")
            dades.carregar_valoracions("dataset/Books/Ratings.csv")
        else:
            print("Dataset no implementat")
            return

        if method == "simple":
            recomanador = RecomanadorSimple(dades)
        elif method == "col·laboratiu":
            recomanador = RecomanadorCol·laboratiu(dades)
        elif method == "contingut":
            recomanador = RecomanadorContingut(dades)
        else:
            print("Mètode no implementat")
            return

        desa_amb_pickle(recomanador, dataset, method)

    while True:
        accio = input("\nAcció? (recomanar/avaluar/sortir): ").lower()
        if accio == "sortir":
            break
        elif accio == "recomanar":
            try:
                user_id = int(input("ID d'usuari: "))
                recomanacions = recomanador.recomana(user_id, 5)
                if not recomanacions:
                    print("No hi ha recomanacions disponibles.")
                else:
                    print(f"\nTop 5 recomanacions per {user_id}:")
                    for idx, (item, puntuacio) in enumerate(recomanacions, 1):
                        print(f"{idx}. {item.get_titol()} ({item.get_categoria()}) - {puntuacio:.2f}")
                        if hasattr(item, "get_info"):
                            print(f"   → {item.get_info()}")
            except ValueError:
                print("ID d'usuari no vàlid.")
        elif accio == "avaluar":
            try:
                user_id = int(input("ID d'usuari: "))
                avaluador = Avaluador(recomanador, recomanador._dades)
                prediccions, reals, mae, rmse = avaluador.avalua(user_id)

                print("\nPrediccions:")
                for item, pred in prediccions[:5]:
                    print(f"- {item.get_titol()} → Predicció: {pred:.2f}")

                print("\nValoracions reals:")
                for item, val in reals[:5]:
                    print(f"- {item.get_titol()} → Valoració: {val:.2f}")

                print(f"\nMAE: {mae:.4f}")
                print(f"RMSE: {rmse:.4f}")
            except ValueError:
                print("ID d'usuari no vàlid.")

if __name__ == "__main__":
    main()