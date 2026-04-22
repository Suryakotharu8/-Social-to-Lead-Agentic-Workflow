import os
import re
import random
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rag import get_kb_context, get_relevant_context
from tools import mock_lead_capture

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: List
    intent: str
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    lead_captured: bool
    awaiting: str | None

# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.3,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_history(messages: list) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Agent: {m.content}")
    return "\n".join(lines) if lines else "(none)"


def extract_email(text: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


KNOWN_PLATFORMS = [
    "youtube", "instagram", "tiktok", "twitter", "facebook",
    "twitch", "linkedin", "snapchat", "pinterest", "reels", "x"
]

def extract_platform(text: str) -> str | None:
    lower = text.lower().strip()
    if lower == "x":
        return "Twitter"
    for p in KNOWN_PLATFORMS:
        if p in lower:
            return "Twitter" if p == "x" else p.capitalize()
    return None


def extract_name(text: str) -> str | None:
    patterns = [
        r"(?:my name is|i am|i'm|call me|it's|its|name['\s]*s?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            stop_words = {"the", "a", "an", "not", "just", "very", "really", "here",
                          "there", "interested", "basic", "pro", "plan"}
            if candidate.lower().split()[0] not in stop_words:
                return candidate.title()

    intent_words = ["want", "need", "like", "subscribe", "sign", "buy", "try",
                    "interested", "plan", "get", "start", "use", "have", "know", "tell"]
    candidate = text.strip()
    words = candidate.split()
    if (
        2 <= len(words) <= 4
        and not extract_email(candidate)
        and not extract_platform(candidate)
        and not any(c in candidate for c in ["@", "http", ".com", "?", "!"])
        and not any(w in candidate.lower() for w in intent_words)
    ):
        return candidate.title()
    return None


def _try_extract_all_fields(text: str, state: AgentState) -> AgentState:
    if not state.get("lead_email"):
        email = extract_email(text)
        if email:
            state["lead_email"] = email

    if not state.get("lead_platform"):
        platform = extract_platform(text)
        if platform:
            state["lead_platform"] = platform

    if not state.get("lead_name"):
        text_clean = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+[.][a-zA-Z0-9-.]+', '', text)
        for p in KNOWN_PLATFORMS:
            text_clean = re.sub(rf'\b{p}\b', '', text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r'\b(and|email|id|is|my)\b', '', text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        name = extract_name(text_clean) or extract_name(text)
        if name:
            state["lead_name"] = name

    return state


def _next_missing_field(state: AgentState) -> str | None:
    if not state.get("lead_name"):
        return "name"
    if not state.get("lead_email"):
        return "email"
    if not state.get("lead_platform"):
        return "platform"
    return None


# ── Node 1: classify_node ─────────────────────────────────────────────────────

def classify_node(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return state

    if state.get("awaiting") and not state.get("lead_captured"):
        state["intent"] = "high_intent"
        return state

    llm = get_llm()

    classification_prompt = f"""You are an intent classifier for a SaaS sales agent.

Classify the user's latest message into EXACTLY one of these three labels:
  casual       → greetings, small talk, unrelated topics
  inquiry      → questions about product, pricing, features, or policies
  high_intent  → user wants to sign up, try, buy, subscribe, or start a plan

Conversation history:
{_format_history(state["messages"][:-1])}

Latest user message: "{last_human.content}"

IMPORTANT: Reply with ONLY one word. No explanation. No punctuation.
Your answer must be exactly one of: casual, inquiry, high_intent"""

    result = llm.invoke([HumanMessage(content=classification_prompt)])
    raw = result.content.strip().lower().split()[0] if result.content.strip() else ""

    if "high_intent" in raw or raw == "high":
        intent = "high_intent"
    elif "inquiry" in raw:
        intent = "inquiry"
    elif "casual" in raw:
        intent = "casual"
    else:
        intent = "inquiry"

    if (
        intent != "high_intent"
        and not state.get("lead_captured")
        and (state.get("lead_name") or state.get("lead_email"))
    ):
        intent = "high_intent"

    state["intent"] = intent
    return state


# ── Node 2: rag_node ──────────────────────────────────────────────────────────

def rag_node(state: AgentState) -> AgentState:
    llm = get_llm()
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    user_text = last_human.content if last_human else ""

    relevant_kb = get_relevant_context(user_text)

    system_prompt = f"""You are AutoStream's friendly sales assistant on social media.
AutoStream is a SaaS platform providing automated video editing tools for content creators.

Use ONLY the retrieved knowledge base context below to answer questions. Never fabricate information.

== RETRIEVED CONTEXT ==
{relevant_kb}

== BEHAVIOR ==
- Be warm, concise, and helpful.
- Answer from the knowledge base only.
- If lead has been captured, you may address the user warmly by name.
- If user seems interested in signing up, encourage them warmly.
- Do NOT collect name/email/platform yourself — the system handles that.
- Do NOT call any tools.
"""
    chat_history = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(chat_history)
    state["messages"].append(AIMessage(content=response.content))
    return state


# ── Node 3: lead_flow_node ────────────────────────────────────────────────────

OPENERS = [
    "That's awesome! 🎬 Let me grab a few quick details.\n\nWhat's your full name?",
    "Great choice! Just need a couple of details to get you started.\n\nWhat's your full name?",
    "Perfect! Let me set you up with AutoStream. What's your full name?",
    "Exciting! 🚀 I'll get you set up right away.\n\nCould I get your full name first?",
]

def lead_flow_node(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    user_text = last_human.content if last_human else ""

    # Step 1: Extract ALL fields first
    state = _try_extract_all_fields(user_text, state)

    # Step 2: Check missing AFTER extraction
    missing = _next_missing_field(state)
    state["awaiting"] = missing

    # Step 3: All collected → tool_node
    if missing is None:
        return state

    # Step 4: Prompt for next field
    if missing == "name":
        no_fields_yet = not any([state.get("lead_email"), state.get("lead_platform")])
        reply = random.choice(OPENERS) if no_fields_yet else "Could you share your full name?"

    elif missing == "email":
        name = state.get("lead_name", "there")
        replies = [
            f"Nice to meet you, {name}! 😊 What's your email address?",
            f"Great, {name}! What email should we use to reach you?",
            f"Got it, {name}! Could you share your email address?",
        ]
        reply = random.choice(replies)

    elif missing == "platform":
        prev_platform_asks = [m for m in state["messages"] if isinstance(m, AIMessage)
                               and "which platform" in m.content.lower()]
        if prev_platform_asks:
            reply = "Sorry, I didn't catch that. Could you specify a platform like YouTube, Instagram, or Twitter?"
        else:
            reply = "Which platform do you mainly create content for? (YouTube, Instagram, TikTok, etc.)"

    state["messages"].append(AIMessage(content=reply))
    return state


# ── Node 4: tool_node ─────────────────────────────────────────────────────────

def tool_node(state: AgentState) -> AgentState:
    if state.get("lead_captured"):
        return state

    name     = state.get("lead_name")
    email    = state.get("lead_email")
    platform = state.get("lead_platform")

    if not all([name, email, platform]):
        state["awaiting"] = _next_missing_field(state)
        return state

    result = mock_lead_capture(name, email, platform)
    state["lead_captured"] = True
    state["awaiting"] = None
    state["messages"].append(AIMessage(content=result))
    return state


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> Literal["rag_node", "lead_flow_node"]:
    if state.get("lead_captured"):
        return "rag_node"
    if state["intent"] == "high_intent":
        return "lead_flow_node"
    return "rag_node"


def route_after_lead_flow(state: AgentState) -> Literal["tool_node", END]:
    if state.get("awaiting") is None and not state.get("lead_captured"):
        return "tool_node"
    return END


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("classify",       classify_node)
    builder.add_node("rag_node",       rag_node)
    builder.add_node("lead_flow_node", lead_flow_node)
    builder.add_node("tool_node",      tool_node)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify",       route_after_classify)
    builder.add_conditional_edges("lead_flow_node", route_after_lead_flow)
    builder.add_edge("rag_node",  END)
    builder.add_edge("tool_node", END)

    return builder.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def create_initial_state() -> AgentState:
    return AgentState(
        messages=[],
        intent="casual",
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        awaiting=None,
    )


def chat(user_input: str, state: AgentState) -> tuple[str, str, AgentState]:
    graph = build_graph()
    state["messages"].append(HumanMessage(content=user_input))
    new_state = graph.invoke(state)
    last_ai = next(
        (m for m in reversed(new_state["messages"]) if isinstance(m, AIMessage)), None
    )
    reply  = last_ai.content if last_ai else "Sorry, something went wrong."
    intent = new_state.get("intent", "inquiry")
    return reply, intent, new_state