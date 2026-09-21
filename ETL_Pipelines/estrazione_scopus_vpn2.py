import os
import time
import json
import requests
from requests.exceptions import RequestException
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ==========================================
# Configurazione Ambiente e Sicurezza
# ==========================================
load_dotenv()

API_KEY = os.getenv("SCOPUS_API_KEY")
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD")) 
BACKUP_FILE = "abstracts_backup.json"

HEADERS = {
    "Accept": "application/json"
}

def fetch_abstract_from_scopus(eid, max_retries=3):
    """Esegue la chiamata HTTP all'API di Scopus con logica di retry in caso di timeout."""
    url = f"https://api.elsevier.com/content/abstract/eid/{eid}?apiKey={API_KEY}&view=FULL"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                coredata = data.get("abstracts-retrieval-response", {}).get("coredata", {})
                return coredata.get("dc:description"), 200
            else:
                return None, response.status_code
                
        except RequestException:
            print(f"[!] Timeout di rete (Tentativo {attempt + 1}/{max_retries}). Nuovo tentativo in 3 secondi...")
            time.sleep(3)
            
    return None, "NetworkError"

def main():
    # 1. Ripristino stato precedente per fault tolerance
    results = {}
    if os.path.exists(BACKUP_FILE):
        try:
            with open(BACKUP_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Backup locale rilevato: {len(results)} abstract già processati.")
        except json.JSONDecodeError:
            print("File di backup illeggibile, inizializzazione da zero.")
            
    # 2. Connessione al database e recupero EID mancanti
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    query_get_eids = """
    MATCH (d:Department)<-[:WORKS_IN]-(a:Author)-[:WRITE]->(p:Document)
    WHERE d.Department_abbr IN ['DIEF', 'DISMI', 'FIM'] 
      AND p.IdScopus IS NOT NULL AND p.IdScopus <> "" AND p.abstract IS NULL
    RETURN DISTINCT p.IdScopus AS eid
    """
    
    with driver.session() as session:
        result = session.run(query_get_eids)
        eids_to_download = [record["eid"] for record in result if record["eid"] not in results]
        
    print(f"Documenti da scaricare in questa sessione: {len(eids_to_download)}")
    
    # 3. Estrazione sequenziale
    for count, eid in enumerate(eids_to_download, 1):
        abstract, status = fetch_abstract_from_scopus(eid)
        
        if abstract:
            results[eid] = abstract
            print(f"[{count}/{len(eids_to_download)}] Abstract recuperato: {eid}")
            
            # Salvataggio incrementale
            if count % 10 == 0:
                with open(BACKUP_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)
        else:
            print(f"[{count}/{len(eids_to_download)}] Errore {status} per l'EID {eid}")
            
        time.sleep(1.0) # Rate limiting
        
    # Chiusura e salvataggio finale
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    driver.close()
    print(f"\nOperazione conclusa. Totale abstract salvati: {len(results)}.")

if __name__ == "__main__":
    main()