from sklearn.feature_extraction.text import TfidfVectorizer


class recomenador_contingut(Recomenador):
    def __init__(self, dades: Dades):
        super().__init__(dades)
        if isinstance(dades, DadesPelis):
            self.valoracio_maxima= 5.0
        else:
            self.valoracio_maxima = 10.0
        self._prepara_items()

    def prepara_items(self):

        self.caractersistiques = []
        items = sorted(self._dades._items.items(), key=lambda x: self._dades._item_id_to_idx[x[0]])
        
        for item_id, item in items:
            if isinstance(item, Peli):
                genres = item._genere.replace('|', ' ')
                self.caracteristiques.append(genres)
            elif isinstance(item, Llibre):
                features = f"{item._autor} {item._editorial} {item._any}"
                self.caracteristiques.append(features)
        
        if not self.caracteristiques:
            self.vectors_items = None
        else:
            self.vectoritzador = TfidfVectorizer()
            self.vectors_item = self.vectoritzador.fit_transform(self.caracteristiques)

        