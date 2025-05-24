sequenceDiagram
    participant Usuari as Usuari
    participant Main as Programa Principal
    participant DadesPelis as DadesPelis
    participant RecomanadorSimple as RecomanadorSimple
    participant UsuariObj as Usuari
    participant Items as Items

    Usuari->>Main: Executa el programa
    Main->>Usuari: "Pelis o llibres?"
    Usuari->>Main: Respon "pelis"

    Main->>DadesPelis: new DadesPelis(path)
    DadesPelis->>DadesPelis: carregar_usuaris(path_usuaris)
    DadesPelis->>DadesPelis: carregar_items(path_items)
    DadesPelis->>DadesPelis: carregar_valoracions(path_valoracions)
    Main->>RecomanadorSimple: new RecomanadorSimple(dades)

    Usuari->>Main: Introdueix user_id i n
    Main->>RecomanadorSimple: recomana(user_id, n)

    alt Usuari no existeix
        RecomanadorSimple->>DadesPelis: get_usuari(user_id)
        DadesPelis-->>RecomanadorSimple: None
        RecomanadorSimple-->>Main: Llança ValueError
        Main-->>Usuari: "Error: Usuari no existeix"
    else Usuari existeix
        RecomanadorSimple->>DadesPelis: get_usuari(user_id)
        DadesPelis-->>RecomanadorSimple: UsuariObj
        RecomanadorSimple->>UsuariObj: Obtenir valoracions
        UsuariObj-->>RecomanadorSimple: Dict(Tuple(int, int), float)
        RecomanadorSimple->>DadesPelis: get_items()
        DadesPelis-->>RecomanadorSimple: Llista d'items

        loop Per cada item a la llista
            RecomanadorSimple->>Items: get_id()
            Items-->>RecomanadorSimple: id_item
            RecomanadorSimple->>RecomanadorSimple: Calcula puntuació (algorisme simple)
        end

        RecomanadorSimple->>RecomanadorSimple: Filtra items no valorats per l'usuari
        RecomanadorSimple->>RecomanadorSimple: Ordena per puntuació (descendent)
        RecomanadorSimple->>RecomanadorSimple: Selecciona top-n items
        RecomanadorSimple-->>Main: Llista ordenada (top-n)
        Main-->>Usuari: Mostra recomanacions
    end