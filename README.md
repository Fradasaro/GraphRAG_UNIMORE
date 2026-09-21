# 🎓 Assistente GraphRAG - UNIMORE

**Un motore di ricerca ibrido basato su Knowledge Graph e LLM per l'esplorazione avanzata delle pubblicazioni accademiche.**

## 📌 Descrizione del Progetto
I tradizionali sistemi RAG (Retrieval-Augmented Generation) basati puramente su vettori semantici falliscono quando devono rispondere a query quantitative o relazionali (es. *"Quanti paper ha scritto l'autore X nel dipartimento Y?"*). 

Questo progetto supera il limite implementando un'architettura **GraphRAG ibrida**. Sfruttando la topologia di un database a grafo (Neo4j), il sistema è in grado di instradare dinamicamente le domande dell'utente verso il motore di ricerca più adatto, unendo il calcolo vettoriale del coseno per l'analisi semantica e i filtri strutturati Cypher per l'esattezza del dato topologico.

## ⚙️ Architettura e Motori di Ricerca
Il cuore del sistema è un agente LangChain dotato di memoria conversazionale, capace di orchestrare tre diversi motori di retrieval:
1. **Motore Strutturale (Text-to-Cypher):** Estrae dati precisi e conteggi navigando gli archi del grafo (es. relazioni `WORKS_IN` e `WRITE`).
2. **Motore Semantico (Vector Search):** Calcola la similarità vettoriale sugli abstract (OpenAI `text-embedding-3-small` a 1536 dimensioni) per query concettuali.
3. **Motore Ibrido Dinamico:** Utilizza Pydantic per forzare l'LLM a estrarre entità strutturate in formato JSON, iniettandole poi in una singola transazione Cypher che fonde la Vector Search con clausole `WHERE` rigide sul grafo.

## 🛠️ Stack Tecnologico
- **Database:** Neo4j (Graph Database & Vector Index)
- **Orchestrazione:** LangChain, Pydantic
- **LLM & Embeddings:** OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`)
- **Frontend:** Streamlit
- **Linguaggio:** Python
🚀 Guida all'installazione
1. Ripristino del Database
Il cuore del sistema è il grafo Neo4j. Non è necessario eseguire nuovamente l'estrazione da Scopus.
Scarica il backup del database (.dump) da Google Drive
Apri Neo4j Desktop e crea un nuovo DBMS vuoto (Versione consigliata: >= 5.x).
Imposta la password iniziale.
Dal menu del DBMS (i tre puntini ...), seleziona Load from Dump e scegli il file neo4j.dump appena scaricato.
Avvia il database.
2. Configurazione dell'Ambiente
Crea un ambiente virtuale e installa le dipendenze necessarie:
Bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3. Variabili d'Ambiente
Rinomina il file .env.example in .env e inserisci le tue credenziali:
Snippet di codice
OPENAI_API_KEY=la_tua_api_key
NEO4J_PASSWORD=la_tua_password_neo4j
4. Avvio dell'Applicazione
Lancia l'interfaccia grafica basata su Streamlit con il seguente comando:
Bash
streamlit run app.py
Nota architetturale: La cartella /ETL_Pipelines contiene gli script originali utilizzati per la Data Ingestion (chiamate API Scopus, vettorializzazione e bulk upload Cypher). Questi file sono forniti a esclusivo scopo di documentazione e non sono necessari per l'esecuzione del frontend.
👤 Autore
Francesco D'Asaro
Laurea Magistrale in Ingegneria Informatica - Università degli Studi di Modena e Reggio Emilia (UNIMORE)
