import os
import requests
import streamlit as st
from agent import run_agent, agent

@st.cache_data(ttl=3600)  # cache for 1 hour
def get_available_groq_models():
    default_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview"
    ]
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
        if not api_key:
            return default_models
        
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5
        )
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            excluded_keywords = ["whisper", "guard", "safeguard", "embed"]
            valid_models = []
            for m in models_data:
                model_id = m["id"]
                if not any(kw in model_id.lower() for kw in excluded_keywords):
                    valid_models.append(model_id)
            
            if valid_models:
                if "llama-3.3-70b-versatile" in valid_models:
                    valid_models.remove("llama-3.3-70b-versatile")
                    valid_models.insert(0, "llama-3.3-70b-versatile")
                return valid_models
    except Exception:
        pass
    return default_models

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Design tokens — light theme & custom elements
# ─────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Global reset ── */
  *, *::before, *::after {
    border-radius: 0 !important;
    box-shadow: none !important;
  }

  /* ── Base typography & background ── */
  html, body, [class*="css"],
  .stApp, .main, .block-container {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: #1a1410 !important;
    background-color: #faf8f6 !important;
  }

  .stApp {
    background: #faf8f6 !important;
  }

  /* ── Force ALL text in markdown/write elements to be dark ── */
  .stMarkdown, .stMarkdown p, .stMarkdown li,
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
  .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
  .stMarkdown strong, .stMarkdown em, .stMarkdown span,
  [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3,
  [data-testid="stMarkdownContainer"] strong {
    color: #1a1410 !important;
    background: transparent !important;
  }

  /* ── Sidebar Styling ── */
  [data-testid="stSidebar"] {
    background-color: #f4ede8 !important;
    border-right: 1px solid #ddd6cf;
  }
  .sidebar-profile {
    text-align: center;
    padding: 1.5rem 0.5rem;
    border-bottom: 1px solid #ddd6cf;
    margin-bottom: 1.5rem;
  }
  .sidebar-avatar {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
  }
  .sidebar-name {
    margin: 0 !important;
    color: #1a1410 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
  }
  .sidebar-role {
    margin: 0.2rem 0 0.6rem !important;
    font-size: 0.75rem !important;
    color: #cc785c !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
  }
  .sidebar-bio {
    font-size: 0.8rem !important;
    color: #7a6a60 !important;
    line-height: 1.45 !important;
    margin: 0 0 1.2rem !important;
    padding: 0 0.4rem !important;
  }
  .sidebar-links {
    display: flex;
    justify-content: center;
    gap: 12px;
  }
  .sidebar-link {
    font-size: 0.75rem !important;
    color: #cc785c !important;
    text-decoration: none !important;
    font-weight: 600 !important;
    border: 1px solid #cc785c !important;
    padding: 3px 10px !important;
    transition: all 0.15s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
  }
  .sidebar-link:hover {
    background: #cc785c !important;
    color: #ffffff !important;
  }

  /* ── Header ── */
  .site-header {
    border-bottom: 1px solid #ddd6cf;
    padding: 1.5rem 0 1rem;
    margin-bottom: 1.4rem;
  }
  .site-wordmark {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #cc785c;
    display: block;
    margin-bottom: 0.35rem;
  }
  .site-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1410;
    line-height: 1.1;
    margin: 0 0 1.1rem;
  }

  /* ── Pipeline strip ── */
  .pipeline-strip {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 0;
    border: 1px solid #ddd6cf;
    width: fit-content;
  }
  .pipe-step {
    padding: 5px 16px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7a6a60;
    border-right: 1px solid #ddd6cf;
  }
  .pipe-step:last-child { border-right: none; }

  /* ── Section label ── */
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #cc785c;
    display: block;
    margin-bottom: 0.6rem;
    margin-top: 0.8rem;
  }

  /* ── Button column alignment ── */
  [data-testid="column"]:last-child .stButton {
    margin-top: 0 !important;
  }
  [data-testid="column"]:last-child .stButton > button {
    height: 42px !important;
    margin-top: 0 !important;
  }

  /* ── Text input ── */
  .stTextInput > div > div > input {
    background: #fdf1f1 !important;
    border: 1px solid #ddd6cf !important;
    border-radius: 0 !important;
    color: #1a1410 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 0.9rem !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #cc785c !important;
    box-shadow: none !important;
    outline: none !important;
  }
  .stTextInput > div > div > input::placeholder {
    color: #b5a89e !important;
  }
  .stTextInput > label {
    color: #7a6a60 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
  }
  .stTextInput > div { border-radius: 0 !important; }

  /* ── Streamlit UI Overrides ── */
  /* Primary buttons (Run Agent) */
  [data-testid="baseButton-primary"] {
    background: #cc785c !important;
    color: #ffffff !important;
    border: 1px solid #cc785c !important;
    border-radius: 0 !important;
    padding: 0.65rem 1.6rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: background 0.15s !important;
    cursor: pointer !important;
    height: 42px !important;
  }
  [data-testid="baseButton-primary"]:hover {
    background: #b8674d !important;
    border-color: #b8674d !important;
  }
  [data-testid="baseButton-primary"]:active {
    background: #a35840 !important;
  }

  /* Secondary buttons (Presets, Downloads) */
  [data-testid="baseButton-secondary"] {
    background: #ffffff !important;
    color: #7a6a60 !important;
    border: 1px solid #ddd6cf !important;
    border-radius: 20px !important;
    padding: 0.35rem 0.9rem !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    width: auto !important;
    letter-spacing: normal !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
  }
  [data-testid="baseButton-secondary"]:hover {
    border-color: #cc785c !important;
    color: #cc785c !important;
    background: #fdf1f1 !important;
  }

  /* ── Status panel ── */
  .status-panel {
    background: #ffffff;
    border: 1px solid #ddd6cf;
    border-left: 3px solid #cc785c;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.6rem;
  }
  .status-title {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7a6a60;
    margin-bottom: 1rem;
  }
  .step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid #ede8e4;
    font-size: 0.85rem;
  }
  .step-row:last-child { border-bottom: none; }
  .step-row.done   { color: #cc785c !important; }
  .step-row.active { color: #1a1410 !important; }
  .step-row.idle   { color: #b5a89e !important; }
  .step-indicator {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    width: 20px;
    text-align: center;
    flex-shrink: 0;
  }

  /* ── Report header card ── */
  .report-wrap {
    border: 1px solid #ddd6cf;
    border-top: 3px solid #cc785c;
    margin-top: 1rem;
    background: #ffffff;
  }
  .report-header {
    padding: 0.8rem 1.4rem;
    border-bottom: 1px solid #ddd6cf;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .report-badge {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #cc785c;
  }
  .report-topic {
    font-size: 0.82rem;
    color: #7a6a60;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Report body ── */
  .report-body-container {
    border: 1px solid #ddd6cf;
    border-top: none;
    background: #ffffff;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
  }
  .report-body-container p,
  .report-body-container li {
    color: #2e2520 !important;
    font-size: 0.94rem !important;
    line-height: 1.65 !important;
    margin-bottom: 0.25rem !important;
  }
  .report-body-container h1,
  .report-body-container h2,
  .report-body-container h3 {
    color: #1a1410 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.4rem !important;
    line-height: 1.2 !important;
    border: none !important;
    background: transparent !important;
  }
  .report-body-container h1 { font-size: 1.3rem !important; }
  .report-body-container h2 { font-size: 1.12rem !important; }
  .report-body-container h3 { font-size: 1.02rem !important; }
  .report-body-container strong { color: #1a1410 !important; font-weight: 700 !important; }
  .report-body-container ul,
  .report-body-container ol {
    padding-left: 1.4rem !important;
    margin: 0.3rem 0 0.6rem !important;
  }
  .report-body-container a { color: #cc785c !important; text-decoration: underline !important; }
  .report-body-container hr {
    border: none !important;
    border-top: 1px solid #ddd6cf !important;
    margin: 0.8rem 0 !important;
  }
  .report-body-container,
  .report-body-container * {
    background-color: transparent !important;
  }
  .report-body-container .stMarkdown,
  .report-body-container [data-testid="stMarkdownContainer"] {
    color: #2e2520 !important;
  }

  /* ── Source cards ── */
  .source-card {
    background: #ffffff;
    border: 1px solid #ddd6cf;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: all 0.2s ease;
  }
  .source-card:hover {
    border-color: #cc785c;
    background: #fdf1f1;
  }
  .source-index {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #cc785c;
    text-transform: uppercase;
  }
  .source-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #1a1410 !important;
    text-decoration: none !important;
  }
  .source-title:hover {
    color: #b8674d !important;
    text-decoration: underline !important;
  }
  .source-url {
    font-size: 0.76rem;
    color: #7a6a60;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
  }

  /* ── Error block ── */
  .error-block {
    background: #fff5f5;
    border: 1px solid #e8c0c0;
    border-left: 3px solid #cc4444;
    padding: 0.9rem 1.2rem;
    color: #8b2020 !important;
    font-size: 0.88rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 1rem;
  }

  /* ── Hide Streamlit chrome action buttons and footer, keeping sidebar toggle visible ── */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  [data-testid="stAppDeployButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────

if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""

def set_topic(preset_text):
    st.session_state.topic_input = preset_text

# ─────────────────────────────────────────────
# Sidebar & Settings
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-profile">
        <div class="sidebar-avatar">🧑‍💻</div>
        <h3 class="sidebar-name">Muhammad Yasir</h3>
        <p class="sidebar-role">Lead AI Developer</p>
        <p class="sidebar-bio">
            Specializing in LangGraph orchestrators, agentic search systems, and high-performance LLM pipelines.
        </p>
        <div class="sidebar-links">
            <a class="sidebar-link" href="https://github.com/yasirx9" target="_blank">GitHub</a>
            <a class="sidebar-link" href="https://www.linkedin.com/in/yasirx9/" target="_blank">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Research Configuration")
    
    available_models = get_available_groq_models()
    default_idx = 0
    if "llama-3.3-70b-versatile" in available_models:
        default_idx = available_models.index("llama-3.3-70b-versatile")
        
    model_choice = st.selectbox(
        "LLM Model",
        options=available_models,
        index=default_idx,
        help="Select the Groq LLM model to power planning and synthesis."
    )
    
    temp_choice = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Higher values make outputs more creative, lower values more factual."
    )
    
    max_res_choice = st.slider(
        "Max Search Results",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Maximum search results to pull per Tavily query."
    )

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="site-header">
  <span class="site-wordmark">Autonomous Research Agent</span>
  <div class="site-title">Research. Verified.</div>
  <div class="pipeline-strip">
    <div class="pipe-step">01 &colon; Plan</div>
    <div class="pipe-step">02 &colon; Search</div>
    <div class="pipe-step">03 &colon; Read</div>
    <div class="pipe-step">04 &colon; Synthesise</div>
    <div class="pipe-step">05 &colon; Review</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input row & presets
# ─────────────────────────────────────────────

st.markdown('<div class="section-label">Research Query</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])

with col_input:
    topic = st.text_input(
        "Topic",
        placeholder="e.g. How is AI being used in Pakistani universities?",
        label_visibility="collapsed",
        key="topic_input",
    )

with col_btn:
    run = st.button("Run Agent", type="primary")

# Suggested topic presets
st.markdown('<span class="section-label" style="margin-top: 0.5rem; font-size: 0.65rem;">Suggested Topics</span>', unsafe_allow_html=True)
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.button("🇵🇰 AI in Pakistani Universities", type="secondary", use_container_width=True, on_click=set_topic, args=("How is AI being used in Pakistani universities?",))
with col_p2:
    st.button("🌌 Commercial Fusion Breakthroughs", type="secondary", use_container_width=True, on_click=set_topic, args=("What are the recent breakthroughs in commercial nuclear fusion energy?",))
with col_p3:
    st.button("🔒 Post-Quantum Cryptography", type="secondary", use_container_width=True, on_click=set_topic, args=("What is the current status of NIST post-quantum cryptography standardization?",))

st.markdown("<hr style='margin: 1.5rem 0 1rem !important;' />", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Pipeline step definitions
# ─────────────────────────────────────────────

STEPS = [
    ("plan",       "Plan",       "Generating search queries"),
    ("search",     "Search",     "Querying Tavily"),
    ("read",       "Read",       "Extracting article text"),
    ("synthesise", "Synthesise", "Drafting report"),
    ("review",     "Review",     "Fact-checking and finalizing"),
]

# ─────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────

if run:
    if not topic.strip():
        st.markdown(
            '<div class="error-block">ERROR — No query provided. Enter a research topic above.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Clear previous result while running
        st.session_state.research_result = None
        st.session_state.last_topic = topic.strip()
        
        status_placeholder = st.empty()

        def render_status(current_idx: int, done: bool = False):
            rows_html = ""
            for i, (_, short, desc) in enumerate(STEPS):
                if done or i < current_idx:
                    cls = "done";   ind = "+"
                elif i == current_idx:
                    cls = "active"; ind = ">"
                else:
                    cls = "idle";   ind = "-"
                rows_html += (
                    f'<div class="step-row {cls}">'
                    f'<span class="step-indicator">{ind}</span>'
                    f'<span><strong>{short}</strong> &colon; {desc}</span>'
                    f'</div>'
                )
            status_placeholder.markdown(f"""
<div class="status-panel">
  <div class="status-title">Pipeline Progress</div>
  {rows_html}
</div>
""", unsafe_allow_html=True)

        render_status(0)
        
        try:
            step_map = {name: i for i, (name, _, _) in enumerate(STEPS)}
            state_inputs = {
                "topic": topic.strip(),
                "temperature": temp_choice,
                "model": model_choice,
                "max_results": max_res_choice,
            }

            for chunk in agent.stream(state_inputs):
                node_name = list(chunk.keys())[0]
                if node_name in step_map:
                    finished_idx = step_map[node_name]
                    next_idx = finished_idx + 1
                    render_status(
                        next_idx if next_idx < len(STEPS) else len(STEPS),
                        done=(next_idx >= len(STEPS)),
                    )

            render_status(0, done=True)

            final_result = run_agent(
                topic.strip(),
                temperature=temp_choice,
                model=model_choice,
                max_results=max_res_choice
            )
            
            # Store in session state
            st.session_state.research_result = final_result
            
            # Clear progress panel
            status_placeholder.empty()

        except Exception as e:
            render_status(0, done=False)
            st.markdown(
                f'<div class="error-block">ERROR — {str(e)}</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
# Render Results Tab Panel
# ─────────────────────────────────────────────

if st.session_state.research_result is not None:
    final_result = st.session_state.research_result
    display_topic = st.session_state.last_topic
    
    # Create tabs
    tab_report, tab_sources, tab_metadata = st.tabs([
        "📄 Research Report",
        "🔗 Sources & Citations",
        "⚙️ Pipeline Metadata"
    ])

    with tab_report:
        # Report header (HTML card top bar)
        st.markdown(f"""
        <div class="report-wrap">
          <div class="report-header">
            <span class="report-badge">Final Report</span>
            <span class="report-topic">{display_topic}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Report body
        st.markdown('<div class="report-body-container">', unsafe_allow_html=True)
        st.markdown(final_result["final_report"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Download buttons side by side
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="⬇️ Download Markdown Report",
                data=final_result["final_report"],
                file_name=f"research_report_{display_topic.replace(' ', '_')}.md",
                mime="text/markdown",
                key="download_md"
            )
        with col_dl2:
            st.download_button(
                label="⬇️ Download Plain Text Report",
                data=final_result["final_report"],
                file_name=f"research_report_{display_topic.replace(' ', '_')}.txt",
                mime="text/plain",
                key="download_txt"
            )

    with tab_sources:
        st.markdown('<span class="section-label">Retrieved Sources</span>', unsafe_allow_html=True)
        results = final_result.get("search_results", [])
        if results:
            for idx, item in enumerate(results):
                st.markdown(f"""
                <div class="source-card">
                    <span class="source-index">Source #{idx+1}</span>
                    <a class="source-title" href="{item['url']}" target="_blank">{item['title']}</a>
                    <span class="source-url">{item['url']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #7a6a60;'>No sources found.</p>", unsafe_allow_html=True)

    with tab_metadata:
        st.markdown('<span class="section-label">Generated Search Queries</span>', unsafe_allow_html=True)
        for q in final_result.get("search_queries", []):
            st.code(q, language=None)

        st.markdown('<span class="section-label" style="margin-top: 1.5rem;">Pipeline Settings Used</span>', unsafe_allow_html=True)
        st.markdown(f"""
        - **Model**: `{model_choice}`
        - **Temperature**: `{temp_choice}`
        - **Max Results per Query**: `{max_res_choice}`
        """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #ddd6cf;">
    <p style="font-size: 0.8rem; color: #7a6a60; font-family: 'Space Grotesk', sans-serif; margin: 0;">
        Autonomous Research Agent &bull; Handcrafted by <strong>Muhammad Yasir</strong> &bull; Powered by LangGraph & Groq
    </p>
</div>
""", unsafe_allow_html=True)
