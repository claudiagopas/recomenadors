from abc import ABC, abstractmethod
import csv
import numpy as np
from typing import List, Dict, Tuple, Union

# === CLASSES BASE ===
class Usuari:
    def __init__(self, user_id: int, location: str = "", age: Union[int, None] = None):
        self._id = user_id
        self._location = location
        self._age = age

    def get_id(self) -> int:
        return self._id

class Item:
    def __init__(self, item_id: int, titol: str = "", categoria: str = ""):
        self._id = item_id
        self._titol = titol
        self._categoria = categoria

    def get_id(self) -> int:
        return self._id

# === CLASSES ESPECÍFIQUES ===
class Llibre(Item):
    def __init__(self, id: int, nom: str, any: int, autor: str):
        super().__init__(id, nom)
        self._any = any
        self._autor = autor

    def get_any(self) -> int:
        return self._any

    def get_autor(self) -> str:
        return self._autor

class Peli(Item):
    def __init__(self, id: int, nom: str, genere: str):
        super().__init__(id, nom)
        self._genere = genere

    def get_genere(self) -> str:
        return self._genere

# === CLASSE ABSTRACTA DADES ===
# Classe abstracta Dades
class Dades(ABC):
    def __init__(self):
        self._recomanacions = np.array([])
        self._users: List[Usuari] = []
        self._items: List[Item] = []
        self._valoracions: Dict[Tuple[int, int], float] = {}

    def get_usuari(self, user_id: int)  -> Union[Usuari, None]:
        return next((u for u in self._users if u.get_id() == user_id), None)

    def get_item(self, item_id: int) -> Union[Item, None]:
        return next((i for i in self._items if i.get_id() == item_id), None)

    @property
    def recomanacions(self):
        return self._recomanacions

    @property
    def users(self):
        return self._users

    @property
    def items(self):
        return self._items

    @property
    def valoracions(self):
        return self._valoracions

    @abstractmethod
    def carregar_usuaris(self, path: str):
        pass

    @abstractmethod
    def carregar_items(self, path: str):
        pass

    @abstractmethod
    def carregar_valoracions(self, path: str):
        pass

# Subclasse per a llibres
class DadesLlibres(Dades):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def carregar(self, fitxer: str) -> list:
        dades = []
        try:
            with open(fitxer, 'r', encoding='utf8') as csv_file:
                reader = csv.reader(csv_file)
                next(reader)
                for row in reader:
                    dades.append(row)
        except FileNotFoundError:
            print(f"Error: el fitxer '{fitxer}' no s'ha trobat.")
        except IOError:
            print(f"Error: no s'ha pogut llegir el fitxer '{fitxer}'.")
        return dades

    def carregar_usuaris(self, path: str):
        files = self.carregar(path)
        self._users = [Usuari(int(f[0]), f[1], f[2]) for f in files if len(f) >= 3]

    def carregar_items(self, path: str):
        files = self.carregar(path)
        self._items = [Item(f[0], f[1], f[2]) for f in files if len(f) >= 3]

    def carregar_valoracions(self, path: str):
        files = self.carregar(path)
        self._valoracions = {(int(f[0]), f[1]): float(f[2]) for f in files if len(f) >= 3}

# Subclasse per a pel·lícules
class DadesPelis(Dades):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._tags = []
        self._links = []

    def carregar(self, fitxer: str) -> list:
        dades = []
        try:
            with open(fitxer, 'r', encoding='utf8') as csv_file:
                reader = csv.reader(csv_file)
                next(reader)
                for row in reader:
                    dades.append(row)
        except FileNotFoundError:
            print(f"Error: el fitxer '{fitxer}' no s'ha trobat.")
        except IOError:
            print(f"Error: no s'ha pogut llegir el fitxer '{fitxer}'.")
        return dades

    def carregar_usuaris(self, path: str):
        files = self.carregar(path)
        user_ids = set()
        for f in files:
            if len(f) >= 1:
                user_ids.add(int(f[0]))
        self._users = [Usuari(user_id) for user_id in sorted(user_ids)]

    def carregar_items(self, path: str):
        files = self.carregar(path)
        self._items = [Item(int(f[0]), f[1], f[2]) for f in files if len(f) >= 3]

    def carregar_valoracions(self, path: str):
        files = self.carregar(path)
        self._valoracions = {(int(f[0]), int(f[1])): float(f[2]) for f in files if len(f) >= 3}

    def carregar_links(self, path: str):
        self._links = self.carregar(path)

    def carregar_tags(self, path: str):
        self._tags = self.carregar(path)

    @property
    def links(self):
        return self._links

    @property
    def tags(self):
        return self._tags

# Exemple de polimorfisme
def carregar_i_mostrar(dades: Dades):
    dades.carregar_usuaris(dades._path + "Users.csv")
    dades.carregar_items(dades._path + "Items.csv")
    dades.carregar_valoracions(dades._path + "Ratings.csv")
    primer_usuari = dades.get_usuari(dades.users[0].get_id())
    print("Usuari trobat:", primer_usuari.get_id())

# === SISTEMA DE RECOMANACIÓ SIMPLE ===
class Recomanador(ABC):
    def __init__(self, dades: Dades):
        self._dades = dades

    @abstractmethod
    def recomana(self, user_id: int, n: int) -> List[Item]:
        pass

class RecomanadorSimple(Recomanador):
    def __init__(self, dades: Dades, min_vots: int = 3):
        super().__init__(dades)
        self._min_vots = min_vots

    def recomana(self, user_id: int, n: int = 5):
        """
        Genera una llista de n Ã­tems recomanats per a un usuari, ordenats per popularitat,
        excloent Ã­tems ja valorats i amb un mÃ­nim de vots.

        Args:
            user_id (int): ID de l'usuari.
            n (int): Nombre d'Ã­tems a recomanar (mÃ xim).

        Returns:
            List[Item]: Llista dels n Ã­tems mÃ©s populars no valorats.

        Raises:
            ValueError: Si l'usuari no existeix o si n Ã©s negatiu.
        """
        
        # Validar usuari
        if not self._dades.get_usuari(user_id):
            raise ValueError(f"Usuari amb ID {user_id} no trobat")

        # Validar n
        if n < 0:
            raise ValueError("El nombre d'Ã­tems a recomanar (n) no pot ser negatiu")
        
        valoracions = self._dades.valoracions
        items = self._dades.items
        items_valorats = {item_id for (uid, item_id), v in valoracions.items() if uid == user_id}

        # Calcular la mitjana de valoracions per cada Ã­tem
        item_valoracions = {}
        for (uid, item_id), v in valoracions.items():
            if v > 0:
                item_valoracions.setdefault(item_id, []).append(v)

        # Filtrar Ã­tems amb un mÃ­nim de vots
        item_avg_valids = [np.mean(votes) for item, votes in item_valoracions.items() if len(votes) >= self._min_vots]
        if not item_avg_valids:
            print("No hi ha prou dades per fer recomanacions.")
            return []

        avg_global = np.mean(item_avg_valids)

        #Calcular puntuacions
        puntuacions = []
        for item in items:
            item_id = item.get_id()
            if item_id in items_valorats:
                continue
            vots = item_valoracions.get(item_id, [])
            num_vots = len(vots)
            if num_vots < self._min_vots:
                continue
            avg_item = np.mean(vots)
            score = (num_vots / (num_vots + self._min_vots)) * avg_item + (self._min_vots / (num_vots + self._min_vots)) * avg_global
            puntuacions.append((item, score))
        puntuacions.sort(key=lambda x: x[1], reverse=True)
        
        if not puntuacions:
            print("No hi ha Ã­tems no valorats amb prou vots per recomanar.")
            return []
        
        return puntuacions[:n]


# === MAIN ===
def main():
    print("=== INICI DEL SISTEMA DE RECOMANACIÓ ===")

    tipus_dades = input("Selecciona el tipus de dades (llibres/pelis): ").strip().lower()
    metode = input("Selecciona el mètode de recomanació (simple): ").strip().lower()

    if tipus_dades not in ["llibres", "pelis"]:
        print("Tipus de dades no vàlid.")
        return

    if metode != "simple":
        print("De moment només està implementat el mètode 'simple'.")
        return

    # Carreguem les dades segons el tipus
    if tipus_dades == "llibres":
        dades = DadesLlibres("carpeta_books/")
        dades.carregar_usuaris("carpeta_books/Users.csv")
        dades.carregar_items("carpeta_books/Books.csv")
        dades.carregar_valoracions("carpeta_books/Ratings.csv")
    else:
        dades = DadesPelis("carpeta_movies/")
        dades.carregar_usuaris("carpeta_movies/ratings.csv")
        dades.carregar_items("carpeta_movies/movies.csv")
        dades.carregar_valoracions("carpeta_movies/ratings.csv")
        dades.carregar_links("carpeta_movies/links.csv")
        dades.carregar_tags("carpeta_movies/tags.csv")

    # Inicialitzem el recomanador
    recomanador = RecomanadorSimple(dades, min_vots=3)

    while True:
        user_input = input("\nIntrodueix l'ID de l'usuari per fer recomanacions (enter per sortir): ").strip()
        if user_input == "":
            print("Finalitzant el programa.")
            break

        try:
            user_id = int(user_input)
        except ValueError:
            print("ID no vàlid. Introdueix un número d'usuari.")
            continue

        usuari = dades.get_usuari(user_id)
        if usuari is None:
            print("Usuari no trobat.")
            continue

        recoms = recomanador.recomana(user_id, n=5)
        if not recoms:
            print("No s'han trobat recomanacions per aquest usuari.")
        else:
            print(f"\nRecomanacions per a l'usuari {user_id}:")
            for item, score in recoms:
                print(f"- Títol: {item._titol}, Categoria: {item._categoria}, Puntuació: {score:.2f}")

if __name__ == "__main__":
    main()
