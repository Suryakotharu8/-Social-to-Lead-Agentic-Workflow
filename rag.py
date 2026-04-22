import json
import os

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")


def load_knowledge_base() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def get_kb_context() -> str:
    """Return full knowledge base as a formatted string (used in system prompt)."""
    kb = load_knowledge_base()
    lines = []

    lines.append(f"Product: {kb['product']}")
    lines.append(f"Description: {kb['description']}\n")

    lines.append("== PRICING PLANS ==")
    for plan in kb["plans"]:
        lines.append(f"\n{plan['name']} — {plan['price']}")
        for feat in plan["features"]:
            lines.append(f"  • {feat}")

    lines.append("\n== COMPANY POLICIES ==")
    for policy in kb["policies"]:
        lines.append(f"  - {policy}")

    lines.append("\n== FAQs ==")
    for faq in kb["faqs"]:
        lines.append(f"  Q: {faq['q']}")
        lines.append(f"  A: {faq['a']}")

    return "\n".join(lines)


def get_relevant_context(query: str) -> str:
    """
    Keyword-based retrieval: returns only KB sections relevant to the query.
    This is the 'retrieval' step in the RAG pipeline — avoids injecting
    the full KB when only a subset is needed.
    Falls back to full KB if no specific section matches.
    """
    kb = load_knowledge_base()
    query_lower = query.lower()
    sections = []

    # Always include product header
    sections.append(f"Product: {kb['product']}\n{kb['description']}")

    # Pricing / plan keywords
    pricing_keywords = ["price", "pricing", "plan", "cost", "basic", "pro", "month", "pay", "cheap", "expensive", "$"]
    if any(k in query_lower for k in pricing_keywords):
        lines = ["== PRICING PLANS =="]
        for plan in kb["plans"]:
            lines.append(f"\n{plan['name']} — {plan['price']}")
            for feat in plan["features"]:
                lines.append(f"  • {feat}")
        sections.append("\n".join(lines))

    # Policy keywords
    policy_keywords = ["refund", "cancel", "support", "policy", "policies", "help", "24/7"]
    if any(k in query_lower for k in policy_keywords):
        lines = ["== COMPANY POLICIES =="]
        for policy in kb["policies"]:
            lines.append(f"  - {policy}")
        sections.append("\n".join(lines))

    # Feature / FAQ keywords
    faq_keywords = ["feature", "caption", "4k", "resolution", "ai", "edit", "video", "platform", "youtube", "instagram", "tiktok"]
    if any(k in query_lower for k in faq_keywords):
        lines = ["== FAQs =="]
        for faq in kb["faqs"]:
            lines.append(f"  Q: {faq['q']}")
            lines.append(f"  A: {faq['a']}")
        sections.append("\n".join(lines))

    # Fallback: full KB if nothing specific matched
    if len(sections) == 1:
        return get_kb_context()

    return "\n\n".join(sections)
