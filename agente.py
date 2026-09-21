import os
import warnings
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ==========================================
# Configurazione Ambiente e Logging
# ==========================================
# Disabilitazione dei warning e dei log di basso livello del driver Neo4j 
# per mantenere l'output del terminale pulito.
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
load_dotenv()
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD")) 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

driver = GraphDatabase.driver(URI, auth=AUTH)

# Inizializzazione del modello di embedding e dell'LLM per la traduzione Cypher
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
llm_per_cypher = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})

# ==========================================
# 1. Definizione dei Retriever di Base
# ==========================================
vector_retriever = VectorRetriever(
    driver=driver, index_name="abstract_embeddings", embedder=embedder, return_properties=["IdScopus", "abstract"]
)

text2cypher_retriever = Text2CypherRetriever(driver=driver, llm=llm_per_cypher)

# ==========================================
# 2. Funzioni Wrapper per i Tool
# ==========================================
def usa_vector(query: str) -> str:
    """Esegue una ricerca semantica basata sulla similarità vettoriale degli abstract."""
    res = vector_retriever.search(query_text=query, top_k=3)
    return "\n".join([str(item.content) for item in res.items])

def usa_text2cypher(query: str) -> str:
    """Traduce la query in linguaggio naturale in Cypher per interrogazioni strutturali topologiche."""
    query_arricchita = f"""
Genera ESCLUSIVAMENTE una query Cypher valida per Neo4j in base alla richiesta dell'utente.


1. Formato Autori: La proprietà Name vuole SOLO la prima lettera maiuscola (es. 'Laura'). La proprietà Surname DEVE essere TUTTA IN MAIUSCOLO (es. 'PO').
2. Formato Dipartimenti: Usa sempre l'operatore IN con la sigla (es. WHERE dep.Department_abbr IN ['DIEF']).
3. Conteggi: Usa count(DISTINCT d) per i paper, count(DISTINCT a) per gli autori. Nessun limite.



Richiesta: Quanti paper ha scritto Laura Po?
Cypher: MATCH (d:Document)<-[:WRITE]-(a:Author {{Name: 'Laura', Surname: 'PO'}}) RETURN count(DISTINCT d)

Richiesta: Qual è il numero totale di paper del dipartimento DIEF?
Cypher: MATCH (d:Document)<-[:WRITE]-(a:Author)-[:WORKS_IN]->(dep:Department) WHERE dep.Department_abbr IN ['DIEF'] RETURN count(DISTINCT d)

Richiesta: In quale dipartimento lavora Francesco Guerra?
Cypher: MATCH (a:Author {{Name: 'Francesco', Surname: 'GUERRA'}})-[:WORKS_IN]->(dep:Department) RETURN dep.Department_abbr



{query}

"""
    try:
        res = text2cypher_retriever.search(query_text=query_arricchita)
        cypher_generato = ""
        if hasattr(res, "metadata") and res.metadata and "cypher" in res.metadata:
            cypher_generato = f"CYPHER ESEGUITO:\n{res.metadata['cypher']}\n"
        
        if not res.items or (len(res.items) == 1 and str(res.items[0].content).strip() == "{}"):
             return f"{cypher_generato}\nRisultato DB: 0. Nessuna corrispondenza."
             
        risultati_db = "\n".join([str(item.content) for item in res.items])
        return f"{cypher_generato}\nRisultati:\n{risultati_db}"
        
    except Exception as e:
        return f"Errore di sintassi Cypher: {str(e)}"

def usa_fulltext(query: str) -> str:
    """Ricerca full-text classica basata su corrispondenze esatte di parole chiave."""
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
# 3. Retriever Ibrido Strutturale-Semantico
# ==========================================
class EntitaRicerca(BaseModel):
    """Schema Pydantic per forzare l'estrazione deterministica dei parametri."""
    dipartimento: str = Field(description="La sigla esatta del dipartimento (es. DIEF, FIM, DISMI, etc.). Ritorna stringa vuota se non specificato.")
    argomento: str = Field(description="L'argomento o concetto di ricerca (es. intelligenza artificiale, big data).")

def usa_ibrido_dinamico(query: str) -> str:
    """Implementa la logica ibrida: estrazione entità (LLM) + Ricerca Vettoriale + Filtro Topologico."""
    estrattore_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(EntitaRicerca)
    entita = estrattore_llm.invoke(query)
    
    argomento = entita.argomento
    dipartimento = entita.dipartimento.upper() if entita.dipartimento else ""
    
    clausola_where = ""
    if dipartimento:
         clausola_where = f"WHERE dep.Department_abbr = '{dipartimento}'"
    
    query_cypher_dinamica = f"""
    CALL db.index.vector.queryNodes('abstract_embeddings', 5, $embedding)
    YIELD node, score
    MATCH (node)<-[:WRITE]-(a:Author)-[:WORKS_IN]->(dep:Department)
    {clausola_where}
    RETURN node.Title AS titolo, a.Name + ' ' + a.Surname AS autore, dep.Department_abbr AS dip, score
    ORDER BY score DESC LIMIT 5
    """
    
    vettore_ricerca = embedder.embed_query(argomento)
    
    with driver.session() as session:
        result = session.run(query_cypher_dinamica, embedding=vettore_ricerca)
        records = [f"Titolo: {record['titolo']} | Autore: {record['autore']} | Dipartimento: {record['dip']} (Score: {record['score']:.2f})" for record in result]
    
    return "\n".join(records) if records else f"Nessun risultato trovato per l'argomento '{argomento}' nel dipartimento '{dipartimento}'."

# ==========================================
# 4. Registrazione Tool e Inizializzazione Agente
# ==========================================
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
prompt_sistema = "Sei l'assistente virtuale del Knowledge Graph di UNIMORE. Rispondi in italiano in base ai dati estratti dai tool."

orchestratore = create_agent(model=llm_agente, tools=tools, system_prompt=prompt_sistema)

def interroga_agente(storico_messaggi: list, callbacks=None) -> str:
    """Interfaccia di comunicazione tra l'app Streamlit e l'agente LangChain, comprensiva di gestione storico."""
    config = {"callbacks": callbacks} if callbacks else {}
    risultato = orchestratore.invoke({"messages": storico_messaggi}, config=config)
    return risultato["messages"][-1].content

# Entry point per i test a riga di comando
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