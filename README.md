# Assistente GraphRAG - UNIMORE

Questo progetto implementa un sistema di Retrieval-Augmented Generation (RAG) basato su un Knowledge Graph (Neo4j) per l'interrogazione semantica e topologica delle pubblicazioni accademiche. 

##  Guida all'installazione per la riproduzione del progetto

### 1. Ripristino del Database
Il cuore del sistema è il grafo Neo4j. Non è necessario eseguire nuovamente l'estrazione da Scopus.
- Apri Neo4j Desktop e crea un nuovo database (DBMS) vuoto (Versione consigliata: 5.x).
- Imposta la password iniziale.
- Clicca sui tre puntini `...` a fianco del nuovo DBMS, seleziona **Load from Dump** e seleziona il file `neo4j.dump` presente nella cartella `/database`.
- Avvia il database.

### 2. Configurazione dell'Ambiente
Crea un ambiente virtuale e installa le dipendenze necessarie:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Chiavi API e Sicurezza
Rinomina il file `.env.example` in `.env` e compila le variabili con la tua password di Neo4j e la tua chiave API di OpenAI.

### 4. Avvio dell'Applicazione
Lancia l'interfaccia grafica basata su Streamlit con il seguente comando:
```bash
streamlit run app.py
```

*Nota: La cartella `/ETL_Pipelines` contiene gli script originali utilizzati per la Data Ingestion (API Scopus, generazione embedding vettoriali e caricamento massivo in Cypher). Questi file sono forniti a solo scopo di documentazione architetturale e non devono essere eseguiti per avviare il sistema.*