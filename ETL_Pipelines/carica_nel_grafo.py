import os
import json
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ==========================================
# Configurazione Ambiente e Sicurezza
# ==========================================
load_dotenv()

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD"))
BACKUP_FILE = "abstracts_backup.json"

def pulisci_testo(testo):
    """Rimuove la formattazione HTML e normalizza la spaziatura."""
    if not testo:
        return ""
    testo_pulito = re.sub(r'<[^>]+>', '', testo)
    testo_pulito = " ".join(testo_pulito.split())
    return testo_pulito

def update_graph_with_abstracts():
    print(f"Lettura dei dati in corso ({BACKUP_FILE})...")
    try:
        with open(BACKUP_FILE, "r", encoding="utf-8") as f:
            abstracts_data = json.load(f)
        print(f"Elementi pronti per il processing: {len(abstracts_data)}")
    except FileNotFoundError:
        print(f"Errore: File {BACKUP_FILE} non trovato.")
        return

    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    # Query Cypher per il bulk update
    update_query = """
    UNWIND $batch AS record
    MATCH (p:Document {IdScopus: record.eid})
    SET p.abstract = record.abstract
    """
    
    # Parsing e sanitizzazione del testo
    print("Sanitizzazione del testo in corso...")
    batch = []
    for eid, text in abstracts_data.items():
        testo_pulito = pulisci_testo(text)
        batch.append({"eid": eid, "abstract": testo_pulito})
    
    # Esecuzione transazione Cypher
    print("Scrittura sul database Neo4j...")
    with driver.session() as session:
        session.run(update_query, batch=batch)
        
    driver.close()
    print("Integrazione completata. Il Knowledge Graph è stato aggiornato.")

if __name__ == "__main__":
    update_graph_with_abstracts()