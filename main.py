import streamlit as st
from graph import chat, create_initial_state

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoStream AI Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d2f3e;
    }

    /* Header */
    .agent-header {
        background: linear-gradient(135deg, #6c63ff 0%, #3ecfcf 100%);
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .agent-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .agent-header p  { color: rgba(255,255,255,0.8); margin: 4px 0 0; font-size: 0.9rem; }

    /* Intent badge */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 4px;
        letter-spacing: 0.5px;
    }
    .badge-casual      { background: #2d4a3e; color: #4ade80; border: 1px solid #4ade80; }
    .badge-inquiry     { background: #1e3a5f; color: #60a5fa; border: 1px solid #60a5fa; }
    .badge-high_intent { background: #4a1e1e; color: #f87171; border: 1px solid #f87171; }

    /* Lead card */
    .lead-card {
        background: linear-gradient(135deg, #1e3a2f, #1a2d1e);
        border: 1px solid #4ade80;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .lead-card h4 { color: #4ade80; margin: 0 0 8px; font-size: 0.85rem; }
    .lead-card p  { color: #cbd5e1; margin: 2px 0; font-size: 0.8rem; }

    /* Summary card */
    .summary-card {
        background: linear-gradient(135deg, #1a2540, #0f1a35);
        border: 1px solid #6c63ff;
        border-radius: 12px;
        padding: 18px;
        margin-top: 10px;
    }
    .summary-card h3 { color: #a78bfa; margin: 0 0 12px; }
    .summary-card p  { color: #cbd5e1; margin: 4px 0; font-size: 0.9rem; }

    /* Metric boxes */
    .metric-box {
        background: #1a1d27;
        border: 1px solid #2d2f3e;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-box .val { font-size: 1.6rem; font-weight: 700; color: #6c63ff; }
    .metric-box .lbl { font-size: 0.75rem; color: #8892a4; margin-top: 2px; }

    /* Chat area */
    .stChatMessage { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "agent_state"   not in st.session_state: st.session_state.agent_state   = create_initial_state()
if "messages"      not in st.session_state: st.session_state.messages      = []
if "intent_log"    not in st.session_state: st.session_state.intent_log    = []
if "captured_leads" not in st.session_state: st.session_state.captured_leads = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 AutoStream Agent")
    st.caption("Powered by **Groq LLaMA 3.1** + **LangGraph**")
    st.divider()

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-box">
            <div class="val">{len(st.session_state.messages) // 2}</div>
            <div class="lbl">Turns</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-box">
            <div class="val">{len(st.session_state.captured_leads)}</div>
            <div class="lbl">Leads</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Lead history log ──────────────────────────────────────────────────────
    st.markdown("### 📋 Captured Leads")
    if st.session_state.captured_leads:
        for i, lead in enumerate(st.session_state.captured_leads, 1):
            st.markdown(f"""<div class="lead-card">
                <h4>🎯 Lead #{i}</h4>
                <p>👤 {lead['name']}</p>
                <p>✉️ {lead['email']}</p>
                <p>📱 {lead['platform']}</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No leads captured yet.")

    st.divider()

    # ── Intent log ────────────────────────────────────────────────────────────
    st.markdown("### 🧠 Intent Log")
    if st.session_state.intent_log:
        for entry in st.session_state.intent_log[-6:]:
            badge_class = f"badge-{entry['intent']}"
            label = entry['intent'].replace("_", " ").upper()
            st.markdown(
                f'<span class="badge {badge_class}">{label}</span> '
                f'<span style="color:#8892a4;font-size:0.75rem">{entry["msg"][:30]}…</span>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No messages yet.")

    st.divider()

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.agent_state    = create_initial_state()
        st.session_state.messages       = []
        st.session_state.intent_log     = []
        st.session_state.captured_leads = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""<div class="agent-header">
    <h1>🎬 AutoStream AI Sales Agent</h1>
    <p>Intelligent lead qualification · RAG-powered knowledge · Real-time intent detection</p>
</div>""", unsafe_allow_html=True)

# Quick start examples
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("👋 What does AutoStream do?", use_container_width=True):
            st.session_state._prefill = "What does AutoStream do?"
            st.rerun()
        if st.button("💰 What are the pricing plans?", use_container_width=True):
            st.session_state._prefill = "What are the pricing plans?"
            st.rerun()
    with c2:
        if st.button("🔒 Do you offer refunds?", use_container_width=True):
            st.session_state._prefill = "Do you offer refunds?"
            st.rerun()
        if st.button("🚀 I want to try the Pro plan!", use_container_width=True):
            st.session_state._prefill = "I want to try the Pro plan for my YouTube channel!"
            st.rerun()
    st.divider()

# Handle prefill from example buttons
prefill_msg = st.session_state.pop("_prefill", None)

# ── Chat history ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Show intent badge on user messages
        if msg["role"] == "user" and i // 2 < len(st.session_state.intent_log):
            intent = st.session_state.intent_log[i // 2]["intent"]
            badge_class = f"badge-{intent}"
            label = intent.replace("_", " ").upper()
            st.markdown(f'<span class="badge {badge_class}">{label}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])

# ── Lead capture summary card ─────────────────────────────────────────────────
# ── Lead capture summary card ─────────────────────────────────────────────────
agent = st.session_state.agent_state
if agent.get("lead_captured") and st.session_state.captured_leads:
    latest = st.session_state.captured_leads[-1]
    st.markdown(f"""<div class="summary-card">
        <h3>✅ Lead Successfully Captured!</h3>
        <p>👤 <b>Name:</b> {latest['name']}</p>
        <p>✉️ <b>Email:</b> {latest['email']}</p>
        <p>📱 <b>Platform:</b> {latest.get('platform', 'N/A')}</p>
        <p style="margin-top:10px;color:#a78bfa;font-size:0.8rem">
            🎯 Lead data has been sent to the CRM pipeline via mock_lead_capture()
        </p>
    </div>""", unsafe_allow_html=True)

# ── Process prefill or chat input ─────────────────────────────────────────────
user_input = prefill_msg or st.chat_input("Ask about pricing, features, or say hi...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            reply, intent, st.session_state.agent_state = chat(user_input, st.session_state.agent_state)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.intent_log.append({"msg": user_input, "intent": intent})

    # Track captured leads in sidebar
    updated = st.session_state.agent_state
    if updated.get("lead_captured"):
        lead = {
            "name":     updated.get("lead_name"),
            "email":    updated.get("lead_email"),
            "platform": updated.get("lead_platform"),
        }
        # Add only if not already in list
        if lead not in st.session_state.captured_leads:
            st.session_state.captured_leads.append(lead)

    st.rerun()
