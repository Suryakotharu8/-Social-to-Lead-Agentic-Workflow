# AutoStream AI Sales Agent

> A Conversational AI agent that converts social media interactions into qualified leads using RAG, intent detection, and tool execution.

Built for **ServiceHive / Inflx** ML Intern Assignment.

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your Groq API Key

→ Sign up free at: https://console.groq.com
→ Create API key → copy it

### 5. Add your API key

Open `graph.py` and replace `"your_groq_key_here"` with your key:

```python
api_key = os.environ.get("GROQ_API_KEY", "your_actual_key_here")
```

### 6. Run the app

```bash
streamlit run main.py
```

Open your browser at: **http://localhost:8501**

---

## 🏗️ Architecture (~200 words)

This agent is built on **LangGraph**, a stateful graph framework from LangChain. LangGraph was chosen over AutoGen because it gives explicit, deterministic control over state transitions — critical when orchestrating a multi-step lead collection flow where premature tool calls must be avoided.

**Graph structure:**
```
[User Input] → classify → [rag_node | lead_flow_node] → [tool_node] → END
```

- **`classify_node`**: Zero-shot intent classification via Groq LLaMA 3.1 with a strict single-word output constraint (`casual | inquiry | high_intent`). Runs every turn; locks to `high_intent` during active collection, resets after lead is captured.

- **`rag_node`**: Handles `casual` and `inquiry` intents. Uses **keyword-based retrieval** (`get_relevant_context()`) to fetch only the relevant KB sections (pricing, policies, or FAQs) rather than injecting the full knowledge base every turn. This is the retrieval step of the RAG pipeline — retrieved context is injected into the system prompt dynamically per query.

- **`lead_flow_node`**: Handles `high_intent`. Uses multi-field extraction to scan each message for name, email, and platform simultaneously before prompting for what's missing. Guides the user through `name → email → platform` sequentially.

- **`tool_node`**: Fires `mock_lead_capture()` only after all three fields are verified present. Safe-guarded against premature execution.

**State** is a plain Python `TypedDict` passed through the graph each turn, retaining full message history, intent, lead fields, and collection stage — enabling 5–6 turn memory without any external store.

---

## 📱 WhatsApp Integration via Webhooks

To deploy this agent on WhatsApp:

1. **Create a Meta App** at [developers.facebook.com](https://developers.facebook.com) and enable the WhatsApp Business API.

2. **Set up a webhook endpoint** (e.g. using FastAPI):
   ```python
   @app.post("/webhook")
   async def webhook(payload: dict):
       user_id = payload["from"]
       user_text = payload["text"]["body"]

       # Load per-user state (from Redis or DB)
       state = load_state(user_id) or create_initial_state()
       reply, intent, new_state = chat(user_text, state)
       save_state(user_id, new_state)

       # Send reply via WhatsApp Cloud API
       send_whatsapp_message(user_id, reply)
   ```

3. **Register webhook** in Meta Developer Console — point it to your deployed server URL.

4. **Verify token**: WhatsApp sends a `GET` challenge on setup; respond with the token to activate.

5. **Persist state per user**: Use Redis or a database keyed by `user_id` (WhatsApp phone number) so each conversation retains memory across turns.

Key difference from web UI: state must be externalized (not in-process) since each webhook call is stateless.

---

## 📁 Project Structure

```
autostream-agent/
├── main.py              # Streamlit web UI + entry point
├── graph.py             # LangGraph state machine (4 nodes: classify, rag, lead_flow, tool)
├── rag.py               # Knowledge base loader + keyword-based retrieval
├── tools.py             # mock_lead_capture() tool
├── knowledge_base.json  # Pricing, features, policies, FAQs
├── requirements.txt
└── README.md
```

---

## 🎯 Agent Capabilities

| Capability | Implementation |
|---|---|
| Intent detection | Zero-shot classification via Groq LLaMA 3.1 (strict single-word output) |
| RAG | Keyword-based retrieval (`get_relevant_context()`) → injected as system prompt context |
| State memory | LangGraph `AgentState` TypedDict across turns |
| Lead collection | Multi-field extraction + deterministic `awaiting` state machine |
| Tool execution | `mock_lead_capture()` called only when all 3 fields verified present |
| Graph nodes | 4 dedicated nodes: `classify`, `rag_node`, `lead_flow_node`, `tool_node` |
| UI | Streamlit — intent badges, lead history sidebar, summary card |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq API key (free at console.groq.com) |

## 📝 Note on LLM Choice

The assignment specifies Gemini 1.5 Flash, GPT-4o-mini, or Claude 3 Haiku. This project was originally built with Gemini 1.5 Flash but switched to **Groq LLaMA 3.1** due to API quota exhaustion from prior projects on the same Google account. The free tier limit was fully consumed, making Gemini unavailable for testing and demo.

Groq was chosen as a drop-in replacement because:
- Free tier with generous limits (14,400 requests/day)
- Same LangChain interface — zero code architecture changes
- Faster inference than Gemini free tier
- No region restrictions

The entire LangGraph architecture, RAG pipeline, intent detection, and tool execution logic remains identical and works with any LLM provider. To switch back to Gemini, simply replace `ChatGroq` with `ChatGoogleGenerativeAI` in `graph.py`.