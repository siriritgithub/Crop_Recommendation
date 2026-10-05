"""
A small "GenAI" farming assistant tab.

This is intentionally lightweight rather than a full vector-DB RAG pipeline:
  1. RETRIEVE: keyword-overlap search over the app's own structured data
     (government schemes + fertilizer guidance) to find the most relevant
     snippets for the user's question.
  2. AUGMENT: those snippets are injected into the LLM prompt as context.
  3. GENERATE: the Anthropic API answers using that context, so answers
     about schemes/fertilizer are grounded in this app's actual data
     rather than the model's general (and possibly outdated) knowledge.

For anything outside that data (general farming questions), the LLM answers
from its own knowledge and the context section is simply omitted.

Requires an Anthropic API key in .streamlit/secrets.toml:
    ANTHROPIC_API_KEY = "sk-ant-..."
Get one at https://console.anthropic.com/settings/keys
"""
import re
from collections import Counter

import requests
import streamlit as st

from govt_schemes import GOVT_SCHEMES, chatbot_answer as _keyword_fallback_answer
from fertilizer_recommender import fertilizer_dict

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a friendly, practical farming assistant for smallholder farmers in India. "
    "Answer clearly and briefly (short paragraphs or bullet points). If context snippets "
    "from government schemes or fertilizer data are provided below, ground your answer in "
    "them and mention the scheme/fertilizer by name. If no relevant context is given, answer "
    "from general agronomy knowledge and say so. Never invent scheme names, benefit amounts, "
    "or application links that aren't in the provided context."
)


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def _build_corpus():
    """Flatten govt schemes + fertilizer dict into (text, source) chunks."""
    chunks = []
    for state, crops in GOVT_SCHEMES.items():
        for crop, schemes in crops.items():
            for s in schemes:
                text = (
                    f"Scheme: {s['name']} | State: {state} | Crop: {crop} | "
                    f"Benefit: {s['benefit']} | Eligibility: {s['eligibility']} | "
                    f"Type: {s['type']} | Apply at: {s['apply']}"
                )
                chunks.append(text)
    for crop, info in fertilizer_dict.items():
        text = (
            f"Fertilizer guidance for {crop}: N={info['N']}kg/ha, P={info['P']}kg/ha, "
            f"K={info['K']}kg/ha. {info['recommendation']} Note: {info['note']} "
            f"Pest control: {info['pesticide']}"
        )
        chunks.append(text)
    return chunks


_CORPUS = _build_corpus()


def retrieve(query: str, top_k: int = 4) -> list:
    """Simple keyword-overlap retrieval -- no embeddings/vector DB needed at this scale."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scored = []
    for chunk in _CORPUS:
        overlap = len(q_tokens & _tokenize(chunk))
        if overlap > 0:
            scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def ask_chatbot_offline(question: str) -> str:
    """No API key configured -- answer with the small built-in keyword
    responder instead of just showing an error. Lower quality than the LLM
    path, but the tab still works with zero setup."""
    return _keyword_fallback_answer(question)


def ask_chatbot(question: str, api_key: str) -> str:
    if not api_key:
        return ask_chatbot_offline(question)

    context_chunks = retrieve(question)
    context_block = ""
    if context_chunks:
        context_block = "Relevant context from the app's data:\n" + "\n".join(
            f"- {c}" for c in context_chunks
        )

    user_message = question if not context_block else f"{context_block}\n\nQuestion: {question}"

    try:
        response = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 500,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(block.get("text", "") for block in data.get("content", []))
    except requests.exceptions.RequestException as e:
        return (
            "⚠️ Couldn't reach the AI service. Check your internet connection and that "
            f"ANTHROPIC_API_KEY in .streamlit/secrets.toml is valid.\n\nDetails: {e}"
        )
