from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np


class Usuari:
    def __init__(self, id: int, dni: str, nom: str):
        self._id = id
        self._dni = dni
        self._nom = nom
        self._valoracions: Dict[int, float] = {}

    def get_id(self) -> int:
        return self._id

    def get_dni(self) -> str:
        return self._dni

    def get_nom(self) -> str:
        return self._nom

    def get_valoracions(self) -> Dict[int, float]:
        return self._valoracions


class Item(ABC):
    def __init__(self, id: int, nom: str):
        self._id = id
        self._nom = nom
        self._valoracions: Dict[int, float] = {}

    def get_id(self) -> int:
        return self._id

    def get_nom(self) -> str:
        return self._nom

    def get_valoracions(self) -> Dict[int, float]:
        return self._valoracions


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


class Dades(ABC):
    def __init__(self):
        self._recomanacions = np.array([])
        self._users: List[Usuari] = []
        self._items: List[Item] = []
        self._valoracions: Dict[Tuple[int, int], float] = {}

    def get_usuari(self, user_id: int) -> Usuari:
        return next((u for u in self._users if u.get_id() == user_id), None)

    def get_item(self, item_id: int) -> Item:
        return next((i for i in self._items if i.get_id() == item_id), None)

    def get_valoracions(self) -> Dict[Tuple[int, int], float]:
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


class DadesLlibres(Dades):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def carregar_usuaris(self, path: str):
        pass  # Implementació específica per llibres

    def carregar_items(self, path: str):
        pass

    def carregar_valoracions(self, path: str):
        pass


class DadesPelis(Dades):
    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def carregar_usuaris(self, path: str):
        pass  # Implementació específica per pelis

    def carregar_items(self, path: str):
        pass

    def carregar_valoracions(self, path: str):
        pass


class Recomanador(ABC):
    def __init__(self, dades: Dades):
        self._dades = dades

    @abstractmethod
    def recomana(self, user_id: int, n: int):
        pass


class RecomanadorSimple(Recomanador):
    def recomana(self, user_id: int, n: int = 5):
        pass  # Implementació del mètode simple


class RecomanadorCol·laboratiu(Recomanador):
    def recomana(self, user_id: int, n: int = 5):
        pass  # Implementació del mètode col·laboratiu


class RecomanadorContingut(Recomanador):
    def recomana(self, user_id: int, n: int = 5):
        pass  # Implementació basada en tf-idf i similitud


class Avaluador:
    def calcular_mae(self, prediccions: Dict[int, float], reals: Dict[int, float]) -> float:
        errors = [abs(prediccions[i] - reals[i]) for i in prediccions if i in reals]
        return sum(errors) / len(errors) if errors else 0.0

    def calcular_rmse(self, prediccions: Dict[int, float], reals: Dict[int, float]) -> float:
        errors = [(prediccions[i] - reals[i]) ** 2 for i in prediccions if i in reals]
        return (sum(errors) / len(errors)) ** 0.5 if errors else 0.0
