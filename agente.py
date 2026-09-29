import os
import warnings
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ==========================================
# Configurazione Ambiente e Logging
# ==========================================
# Soppressione dei warning e dei log di basso livello del driver Neo4j 
# per mantenere l'output della console pulito durante l'esecuzione.
warnings.filterwarnings("ignore")
logging.getLogger("neo4j").setLevel(logging.ERROR)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.tools import Tool
from neo4j_graphrag.retrievers import VectorRetriever, Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from pydantic import BaseModel, Field

# ==========================================
# Inizializzazione Connessione e API
# ==========================================
# Caricamento delle variabili d'ambiente (credenziali Neo4j e chiavi OpenAI)
load_dotenv()
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD")) 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Driver ufficiale per le query dirette al DB Graph
driver = GraphDatabase.driver(URI, auth=AUTH)

# Modello per trasformare il testo in vettori (Semantic Search)
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
# Modello LLM a temperatura 0 (massimo determinismo, nessuna creatività)
llm_per_cypher = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})

# ==========================================
# 1. Definizione dei Retriever di Base (Libreria Standard)
# ==========================================
# Utilizzo il VectorRetriever di base della libreria neo4j-graphrag solo per le 
# ricerche puramente semantiche, dove non servono filtri complessi.
vector_retriever = VectorRetriever(
    driver=driver, index_name="abstract_embeddings", embedder=embedder, return_properties=["IdScopus", "abstract"]
)

text2cypher_retriever = Text2CypherRetriever(driver=driver, llm=llm_per_cypher)

# ==========================================
# 2. Funzioni Wrapper e Retriever Customizzati
# ==========================================
def usa_vector(query: str) -> str:
    """Esegue una ricerca semantica basata sulla similarità vettoriale degli abstract."""
    res = vector_retriever.search(query_text=query, top_k=3)
    return "\n".join([str(item.content) for item in res.items])

# --- INIZIO MOTORE TEXT-TO-CYPHER CUSTOM ---
class ModelloCypher(BaseModel):
    """
    Schema Pydantic: Forza l'LLM a restituire un oggetto strutturato (JSON) 
    contenente ESCLUSIVAMENTE codice Cypher, impedendogli di inserire 
    testo discorsivo che farebbe crashare l'esecuzione Python.
    """
    query_cypher: str = Field(description="La singola query Cypher pronta per essere eseguita sul database")

def usa_text2cypher(query: str) -> str:
    """
    Motore Text2Cypher personalizzato. Bypassa la "black-box" della libreria standard
    per evitare troncamenti nascosti dei risultati (Tool Output Truncation).
    Implementa inoltre la formattazione Python in backend per aggirare la 
    "Formatting Laziness" dell'LLM di fronte a liste lunghe.
    """
    # Prompting XML ingegnerizzato con regole rigide (Few-Shot Prompting)
    prompt = f"""
Genera ESCLUSIVAMENTE una query Cypher valida per Neo4j in base alla richiesta dell'utente.


1. Formato Autori: Name con SOLO prima lettera maiuscola (es. 'Laura'). Surname TUTTO IN MAIUSCOLO (es. 'PO').
2. Formato Dipartimenti: Usa l'operatore IN (es. WHERE dep.Department_abbr IN ['DIEF']).
3. Conteggi (Quanti): Usa SEMPRE E SOLO count(d). È SEVERAMENTE VIETATO usare la parola DISTINCT (NON scrivere MAI count(DISTINCT d)).
4. Liste (Quali/Elenca): Restituisci i titoli (es. RETURN d.Title).
5. Misti (Quanti e Quali): Usa SEMPRE collect(). Sintassi esatta: RETURN count(d) AS Totale, collect(d.Title) AS Titoli



Richiesta: Quanti paper ha scritto Laura Po?
Cypher: MATCH (d:Document)<-[:WRITE]-(a:Author {{Name: 'Laura', Surname: 'PO'}}) RETURN count(d) AS Totale

Richiesta: Quanti e quali paper ha scritto Francesco Guerra?
Cypher: MATCH (d:Document)<-[:WRITE]-(a:Author {{Name: 'Francesco', Surname: 'GUERRA'}}) RETURN count(d) AS Totale, collect(d.Title) AS Titoli


Richiesta utente: {query}
"""
    try:
        # 1. Generazione deterministica della query tramite Pydantic
        generatore_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ModelloCypher)
        risultato_llm = generatore_llm.invoke(prompt)
        cypher_generato = risultato_llm.query_cypher
        
        # 2. Esecuzione diretta sul DB tramite driver nativo Neo4j
        with driver.session() as session:
            result = session.run(cypher_generato)
            
            risultati_formattati = []
            for record in result:
                d = dict(record)
                
                # 3. Offloading Computazionale: È Python a impaginare la lista (Markdown), 
                # sollevando l'Agente IA dal compito per prevenire rifiuti (LLM Laziness).
                if "Titoli" in d and isinstance(d["Titoli"], list):
                    titoli_str = "\n".join([f"- {titolo}" for titolo in d["Titoli"]])
                    totale = d.get("Totale", len(d["Titoli"]))
                    risultati_formattati.append(f"Totale esatto: {totale}\n\nEcco l'elenco completo dei paper:\n{titoli_str}")
                else:
                    risultati_formattati.append(", ".join([f"{k}: {v}" for k, v in d.items()]))
                    
        if not risultati_formattati:
             return f"Nessuna corrispondenza trovata nel database."
             
        testo_finale = "\n\n".join(risultati_formattati)
        
        # Output blindato: Forziamo l'agente a stampare l'elenco pre-formattato
        return f"DATI ESTRATTI CON SUCCESSO:\n{testo_finale}\n\n[REGOLA TASSATIVA PER L'AGENTE: Ricopia esattamente l'elenco puntato qui sopra per l'utente, non riassumerlo e non tagliarlo per nessun motivo]."
        
    except Exception as e:
        return f"Errore di esecuzione Cypher: {str(e)}"

def usa_fulltext(query: str) -> str:
    """Ricerca full-text classica basata su corrispondenze esatte di parole chiave o acronimi."""
    cypher_query = """
    CALL db.index.fulltext.queryNodes("titoli_paper", $testo) YIELD node, score
    RETURN node.Title AS titolo, score
    LIMIT 5
    """
    with driver.session() as session:
        result = session.run(cypher_query, testo=query)
        records = [f"Titolo: {record['titolo']} (Score: {record['score']:.2f})" for record in result]
    return "\n".join(records) if records else "Nessun paper trovato con queste parole chiave."

# ==========================================
# 3. Retriever Ibrido Strutturale-Semantico (AGGIORNATO)
# ==========================================
class EntitaRicerca(BaseModel):
    """
    Schema Pydantic per il motore Ibrido. Costringe l'LLM a mappare la frase 
    dell'utente estrapolando i metadati esatti per costruire la clausola WHERE.
    I parametri non forniti dall'utente ritornano null o stringhe vuote.
    """
    argomento: str = Field(description="L'argomento o concetto di ricerca (es. intelligenza artificiale, big data).")
    dipartimento: str = Field(default="", description="La sigla esatta del dipartimento (es. DIEF, FIM). Ritorna stringa vuota se non specificato.")
    autore: str = Field(default="", description="Il cognome dell'autore (es. GUERRA, PO, ROSSI). Ritorna stringa vuota se non specificato.")
    anno: int = Field(default=None, description="L'anno di pubblicazione (es. 2023). Ritorna null se non specificato.")

def usa_ibrido_dinamico(query: str) -> str:
    """
    Motore Ibrido Dinamico: Unisce la similarità del coseno (Vettori) al 
    filtro topologico e temporale (Cypher) in una singola transazione.
    """
    # 1. Estrazione Strutturata tramite LLM
    estrattore_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(EntitaRicerca)
    entita = estrattore_llm.invoke(query)
    
    argomento = entita.argomento
    dipartimento = entita.dipartimento.upper() if entita.dipartimento else ""
    autore = entita.autore.upper() if entita.autore else ""
    anno = entita.anno
    
    # 2. Routing Dinamico: Costruzione della clausola WHERE solo per i parametri attivi
    condizioni = []
    if dipartimento:
         condizioni.append(f"dep.Department_abbr = '{dipartimento}'")
    if autore:
         condizioni.append(f"a.Surname = '{autore}'")
    if anno:
         condizioni.append(f"toInteger(node.YearOfPublication) = {anno}") 
    
    clausola_where = ""
    if condizioni:
        clausola_where = "WHERE " + " AND ".join(condizioni)
    
    # 3. Intersezione Semantica-Strutturale
    # NB: 'k' è impostato a 100 nell'indice vettoriale per allargare il bacino 
    # e prevenire il problema dei falsi negativi dovuti al "Post-Filtering".
    query_cypher_dinamica = f"""
    CALL db.index.vector.queryNodes('abstract_embeddings', 100, $embedding)
    YIELD node, score
    MATCH (node)<-[:WRITE]-(a:Author)-[:WORKS_IN]->(dep:Department)
    {clausola_where}
    RETURN node.Title AS titolo, a.Name + ' ' + a.Surname AS autore, dep.Department_abbr AS dip, score
    ORDER BY score DESC LIMIT 5
    """
    
    # Generazione dell'embedding per l'argomento cercato
    vettore_ricerca = embedder.embed_query(argomento)
    
    with driver.session() as session:
        result = session.run(query_cypher_dinamica, embedding=vettore_ricerca)
        records = [f"Titolo: {record['titolo']} | Autore: {record['autore']} | Dipartimento: {record['dip']} (Score: {record['score']:.2f})" for record in result]
    
    # Stringa di log per permettere all'Agente di capire quali filtri ha applicato
    filtri_usati = f"[Dipartimento: {dipartimento}] " if dipartimento else ""
    filtri_usati += f"[Autore: {autore}] " if autore else ""
    filtri_usati += f"[Anno: {anno}]" if anno else ""
    
    return "\n".join(records) if records else f"Nessun risultato per '{argomento}' con i filtri: {filtri_usati.strip()}."

# ==========================================
# 4. Registrazione Tool e Inizializzazione Agente LangChain
# ==========================================
# Le 'description' dei Tool sono fondamentali: agiscono come "Prompt di Routing".
# L'Agente le legge per capire autonomamente quale motore attivare in base alla domanda.
tools = [
    Tool(
        name="Ricerca_Strutturale_Cypher",
        func=usa_text2cypher,
        description="USO TASSATIVO E OBBLIGATORIO per conteggi, per interrogare i paper di UN SINGOLO AUTORE, PER SCOPRIRE IN QUALE DIPARTIMENTO LAVORA UN AUTORE e per calcolare CO-AUTORIALITÀ tra dipartimenti."
    ),
    Tool(
        name="Ricerca_Semantica_Vettoriale",
        func=usa_vector,
        description="USO OBBLIGATORIO per cercare argomenti, tematiche o concetti astratti nei paper (es. 'paper sui big data', 'papers focusing on data engineering', 'intelligenza artificiale'). Usa sempre questo tool quando l'utente vuole scoprire di cosa parlano i paper."
    ),
    Tool(
        name="Ricerca_Ibrida_Avanzata",
        func=usa_ibrido_dinamico,
        description="USO OBBLIGATORIO per le domande MISTE. Usalo se e solo se la domanda contiene sia un TEMA/ARGOMENTO (es. machine learning) sia la richiesta di un'ENTITÀ (es. 'chi sono gli autori che...', 'in quale dipartimento...')."
    ),
    Tool(
        name="Ricerca_Esatta_Parola_Chiave",
        func=usa_fulltext,
        description="USO TASSATIVO per la ricerca testuale classica. Usalo SOLO quando l'utente cerca un TITOLO ESATTO di un paper, una parola chiave specifica o una SIGLA alfanumerica precisa (es. 'cerca il paper BLAST2', 'trova l'articolo intitolato...'). Non usare per esplorare concetti o argomenti generici."
    )
]

llm_agente = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt di Sistema: Impone i guardrail per l'Agente Orchestratore.
# Impedisce esplicitamente le allucinazioni (inventare dati) e le omissioni (riassumere liste).
prompt_sistema = """Sei l'assistente virtuale del Knowledge Graph di UNIMORE. 
Rispondi in italiano basandoti ESCLUSIVAMENTE sui dati estratti dai tool.

REGOLA ANTI-ALLUCINAZIONE: Se un tool restituisce "Nessun risultato" o "0", DEVI dire all'utente che non ci sono documenti nel database. È SEVERAMENTE VIETATO inventare titoli, abstract o autori per compiacere l'utente. Attieniti solo ai dati reali.

REGOLA SULLE LISTE: Se un tool ti restituisce un elenco di titoli (es. 20 o 40 paper), DEVI stamparli TUTTI a schermo come elenco puntato. È vietato riassumerli."""

orchestratore = create_agent(model=llm_agente, tools=tools, system_prompt=prompt_sistema)

def interroga_agente(storico_messaggi: list, callbacks=None) -> str:
    """Interfaccia di comunicazione tra l'app Streamlit e l'agente LangChain, comprensiva di gestione storico per risolvere le coreferenze (memoria)."""
    config = {"callbacks": callbacks} if callbacks else {}
    risultato = orchestratore.invoke({"messages": storico_messaggi}, config=config)
    return risultato["messages"][-1].content

# ==========================================
# Esecuzione in locale (CLI Testing)
# ==========================================
if __name__ == "__main__":
    print("\n[Sistema inizializzato: GraphRAG UNIMORE in ascolto...]\n")
    storico_messaggi = []
    
    while True:
        domanda = input("Utente: ")
        if domanda.lower() in ['esci', 'exit', 'quit', 'q']:
            print("Chiusura del processo in corso...")
            break
            
        print("[Elaborazione richiesta...]")
        storico_messaggi.append({"role": "user", "content": domanda}) 
        
        try:
            risultato = orchestratore.invoke({"messages": storico_messaggi})
            risposta_pulita = risultato["messages"][-1].content
            
            print(f"Agente: {risposta_pulita}\n")
            storico_messaggi.append({"role": "assistant", "content": risposta_pulita})
            
        except Exception as e:
            print(f"Errore di esecuzione: {e}\n")
            storico_messaggi.pop()

    driver.close()