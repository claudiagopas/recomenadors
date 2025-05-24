
from abc import ABC, abstractmethod
import csv
import numpy as np
from typing import List, Dict, Tuple, Union

# Classe Usuari
class Usuari:
    def __init__(self, user_id: int, location: str = "", age: Union[int, None] = None):
        self._id = user_id
        self._location = location
        self._age = age

    def get_id(self) -> int:
        return self._id

# Classe Item
class Item:
    def __init__(self, item_id: int, titol: str = "", categoria: str = ""):
        self._id = item_id
        self._titol = titol
        self._categoria = categoria

    def get_id(self) -> int:
        return self._id

# Classe abstracta Dades
class Dades(ABC):
    def __init__(self):
        self._recomanacions = np.array([])
        self._users: List[Usuari] = []
        self._items: List[Item] = []
        self._valoracions: Dict[Tuple[int, int], float] = {}

    def get_usuari(self, user_id: int) -> Union[Usuari, None]:
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

def main():
    print("=== CÀRREGA DE DADES DE LLIBRES ===")
    llibres = DadesLlibres("carpeta_books/")
    llibres.carregar_usuaris("carpeta_books/Users.csv")
    llibres.carregar_items("carpeta_books/Books.csv")
    llibres.carregar_valoracions("carpeta_books/Ratings.csv")

    print(f"Nombre d'usuaris carregats: {len(llibres.users)}")
    print(f"Nombre d'ítems carregats: {len(llibres.items)}")
    print(f"Nombre de valoracions carregades: {len(llibres.valoracions)}")

    print("\n=== CÀRREGA DE DADES DE PEL·LÍCULES ===")
    pelis = DadesPelis("carpeta_movies/")
    pelis.carregar_usuaris("carpeta_movies/ratings.csv")  # ✅ Ara correcte
    pelis.carregar_items("carpeta_movies/movies.csv")
    pelis.carregar_valoracions("carpeta_movies/ratings.csv")
    pelis.carregar_links("carpeta_movies/links.csv")
    pelis.carregar_tags("carpeta_movies/tags.csv")

    print(f"Nombre d'usuaris carregats: {len(pelis.users)}")
    print(f"Nombre d'ítems carregats: {len(pelis.items)}")
    print(f"Nombre de valoracions carregades: {len(pelis.valoracions)}")
    print(f"Nombre de links carregats: {len(pelis.links)}")
    print(f"Nombre de tags carregats: {len(pelis.tags)}")

if __name__ == "__main__":
    main()
