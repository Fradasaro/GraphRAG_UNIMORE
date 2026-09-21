import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

# ==========================================
# Configurazione Ambiente e Sicurezza
# ==========================================
load_dotenv()

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD")) 

def crea_vettori_e_indici():
    # Inizializzazione del modello di embedding OpenAI
    print("Inizializzazione servizio di embedding...")
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Pulizia indici preesistenti
    print("Rimozione indici vettoriali obsoleti...")
    with driver.session() as session:
        session.run("DROP INDEX abstract_embeddings IF EXISTS")

    # Estrazione degli abstract dal grafo
    print("Recupero dati dal database...")
    query_get = """
    MATCH (p:Document)
    WHERE p.abstract IS NOT NULL
    RETURN p.IdScopus AS eid, p.abstract AS testo
    """
    with driver.session() as session:
        result = session.run(query_get)
        documenti = [{"eid": record["eid"], "testo": record["testo"]} for record in result]

    print(f"Trovati {len(documenti)} documenti. Avvio generazione vettori...")

    # Processamento a blocchi (batch) per ottimizzazione chiamate API
    batch_size = 100
    for i in range(0, len(documenti), batch_size):
        batch_docs = documenti[i:i+batch_size]
        testi = [doc["testo"] for doc in batch_docs]
        
        # Generazione embedding
        vettori = embedder.embed_documents(testi)
        
        batch_data = []
        for j, doc in enumerate(batch_docs):
            batch_data.append({
                "eid": doc["eid"],
                "vettore": vettori[j]
            })
        
        # Scrittura dei vettori sui nodi Document
        query_update = """
        UNWIND $batch AS record
        MATCH (p:Document {IdScopus: record.eid})
        SET p.embedding = record.vettore
        """
        with driver.session() as session:
            session.run(query_update, batch=batch_data)
        
        print(f"  -> Salvati {min(i+batch_size, len(documenti))}/{len(documenti)} embedding...")

    # Costruzione dell'indice vettoriale
    print("Creazione del Vector Index in Neo4j...")
    query_index = """
    CREATE VECTOR INDEX abstract_embeddings IF NOT EXISTS
    FOR (p:Document)
    ON (p.embedding)
    OPTIONS {indexConfig: {
     `vector.dimensions`: 1536,
     `vector.similarity_function`: 'cosine'
    }}
    """
    with driver.session() as session:
        session.run(query_index)

    print("Pipeline completata. Il database è ottimizzato per la ricerca vettoriale.")
    driver.close()

if __name__ == "__main__":
    crea_vettori_e_indici()