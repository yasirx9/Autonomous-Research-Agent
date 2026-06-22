import os
from dotenv import load_dotenv

# Load .env for local development; on Streamlit Cloud, secrets are injected via st.secrets
load_dotenv()

# Streamlit Cloud: pull from secrets if env vars not already set
try:
    import streamlit as st
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    pass  # Not running inside Streamlit (e.g., CLI mode) — .env already loaded above

# ─────────────────────────────────────────────
# Part 2 - LLM, Search Client, and the State
# ─────────────────────────────────────────────

from typing_extensions import TypedDict
from typing import List, Dict
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # is a small converter that extracts just the .content string from that object.
from langgraph.graph import StateGraph, START, END

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

class ResearchState(TypedDict):
    topic: str                  # user's original question
    search_queries: List[str]   # The 3 queries the Plan node generates
    search_results: List[Dict]  # A list of dictionaries, each one representing a search result,
    full_text: str              # All the extracted article text combined
    report: str                 # LLM's first draft of the report
    final_report: str           # Reviewed Final Version
    temperature: float          # LLM temperature
    model: str                  # Groq Model name
    max_results: int            # Tavily Max Results per query

# ─────────────────────────────────────────────
# Part 3 - The 5 Nodes
# ─────────────────────────────────────────────

# NODE 1: Plan - turn the topic into 3 search queries
def plan_node(state):
    model_name = state.get("model", "llama-3.3-70b-versatile")
    temp = state.get("temperature", 0.3)
    dynamic_llm = ChatGroq(model=model_name, temperature=temp)
    
    plan_chain = ChatPromptTemplate.from_template(
        "Generate exactly 3 specific search queries for this topic. One per line, no numbering.\n\nTopic: {topic}"
    ) | dynamic_llm | StrOutputParser()
    
    queries = [q.strip() for q in plan_chain.invoke({"topic": state["topic"]}).split("\n") if q.strip()]
    print(f"[PLAN] {queries}")
    return {"search_queries": queries}


# NODE 2: Search - Tavily search for each query
def search_node(state):
    results = []
    max_res = state.get("max_results", 2)
    for q in state["search_queries"]:
        for r in tavily.search(q, max_results=max_res)["results"]:
            results.append({"title": r["title"], "url": r["url"]})
    print(f"[SEARCH] {len(results)} results found")
    return {"search_results": results}


# NODE 3: Read - Tavily extract full text from top URLs
def read_node(state):
    max_res = state.get("max_results", 2)
    limit = max(4, max_res * 2)
    urls = [r["url"] for r in state["search_results"]][:limit]

    extracted = tavily.extract(urls=urls)["results"]
    full_text = "\n\n".join(f"[{r['url']}]\n{r['raw_content'][:1500]}" for r in extracted)

    print(f"[READ] Extracted {len(extracted)} articles")
    return {"full_text": full_text}


# NODE 4: Synthesise - write a report from the extracted text
def synthesise_node(state):
    model_name = state.get("model", "llama-3.3-70b-versatile")
    temp = state.get("temperature", 0.3)
    dynamic_llm = ChatGroq(model=model_name, temperature=temp)
    
    synthesise_chain = ChatPromptTemplate.from_template(
        "Write a short report on '{topic}' using ONLY the source text below. "
        "Include an intro, 3 key findings, and a conclusion.\n\nSources:\n{full_text}"
    ) | dynamic_llm | StrOutputParser()
    
    report = synthesise_chain.invoke({"topic": state["topic"], "full_text": state["full_text"]})
    print("[SYNTHESISE] Report drafted")
    return {"report": report}


# NODE 5: Review - check the report against sources, finalize
def review_node(state):
    model_name = state.get("model", "llama-3.3-70b-versatile")
    temp = state.get("temperature", 0.3)
    dynamic_llm = ChatGroq(model=model_name, temperature=temp)
    
    review_chain = ChatPromptTemplate.from_template(
        "Check this report against the sources. Remove any unsupported claims, "
        "add a 'Sources' list at the end, return the final version.\n\n"
        "Sources:\n{full_text}\n\nReport:\n{report}"
    ) | dynamic_llm | StrOutputParser()
    
    final = review_chain.invoke({"full_text": state["full_text"], "report": state["report"]})
    print("[REVIEW] Final report ready")
    return {"final_report": final}

# ─────────────────────────────────────────────
# Part 4 - Wire the Graph
# ─────────────────────────────────────────────

graph = StateGraph(ResearchState)

for name, fn in [("plan", plan_node), ("search", search_node), ("read", read_node),
                 ("synthesise", synthesise_node), ("review", review_node)]:
    graph.add_node(name, fn)

graph.add_edge(START, "plan")
graph.add_edge("plan", "search")
graph.add_edge("search", "read")
graph.add_edge("read", "synthesise")
graph.add_edge("synthesise", "review")
graph.add_edge("review", END)

agent = graph.compile()

# ─────────────────────────────────────────────
# run_agent() - called by frontend.py
# ─────────────────────────────────────────────

def run_agent(topic: str, temperature: float = 0.3, model: str = "llama-3.3-70b-versatile", max_results: int = 2) -> dict:
    """
    Run the full 5-node research pipeline for a given topic.
    Returns the full result state dict (use result["final_report"] for the report).
    """
    result = agent.invoke({
        "topic": topic,
        "temperature": temperature,
        "model": model,
        "max_results": max_results
    })
    return result


# ─────────────────────────────────────────────
# CLI fallback - run directly for quick testing
# ─────────────────────────────────────────────

if __name__ == "__main__":
    topic = input("Enter research topic: ")
    result = run_agent(topic)
    print("\n\n========== FINAL REPORT ==========\n")
    print(result["final_report"])

