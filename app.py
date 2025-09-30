import json
from typing import Any, Dict, List, Tuple

import streamlit as st
import plotly.express as px

from backend.extraction import extract_text
from backend.preprocessing import preprocess_contract
from backend.analysis import mock_llm_analyze_clauses, chat_with_claude
from backend.scoring import compute_overall_risk_score, summarize_risk_counts
from backend.export import to_json, to_pdf_report

st.set_page_config(page_title="Legal Contract Risk Analyzer", layout="wide")

st.title("GenAI based Legal Contract Analysis & Risk Assessment")

with st.expander("Instructions", expanded=True):
    st.markdown(
        "Upload a contract in PDF/DOCX/TXT. The app extracts text, preprocesses clauses, runs a mock LLM risk analysis, and computes an overall risk score."
    )

uploaded = st.file_uploader("Upload Contract (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"]) 

# Session state for chat
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_context_key" not in st.session_state:
    st.session_state.chat_context_key = None


RISK_TO_SCORE = {"High": 100, "Medium": 50, "Low": 0}


def reset_chat_for_file(file_key: str):
    if st.session_state.chat_context_key != file_key:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Hi! I read your contract. Ask me anything — I’ll explain in simple terms."
            }
        ]
        st.session_state.chat_context_key = file_key


def generate_reply(user_text: str, analyses: List[Dict[str, Any]], counts: Dict[str, int], overall: int) -> str:
    text = user_text.lower()

    # Overall score queries
    if any(k in text for k in ["overall", "score", "summary", "overview"]):
        return (
            f"Big picture: the overall risk score is {overall}/100. "
            f"We see High={counts.get('High',0)}, Medium={counts.get('Medium',0)}, and Low={counts.get('Low',0)} clauses."
        )

    # Clause-specific queries
    import re as _re
    m = _re.search(r"clause\s*(\d+)", text)
    if m:
        cid = int(m.group(1))
        for item in analyses:
            if item.get("clause_id") == cid:
                return (
                    f"Clause {cid} looks {item['risk_level'].lower()} risk. "
                    f"In short: {item['explanation']} "
                    f"Safer version: {item['suggested_rewrite']}"
                )
        return f"I couldn’t find clause {cid}. Try another number."

    # Ask about high risk / priorities
    if any(k in text for k in ["high risk", "top risk", "biggest risk", "prioritize", "priority"]):
        high = [a for a in analyses if a.get("risk_level") == "High"]
        if not high:
            return "Good news: no High‑risk clauses. You can focus on Medium‑risk items for improvements."
        top = high[:3]
        lines = ["These are the riskiest spots to fix first:"]
        for x in top:
            lines.append(f"- Clause {x['clause_id']}: {x['explanation']}")
        return "\n".join(lines)

    # Safer wording suggestions
    if any(k in text for k in ["rewrite", "safer", "alternative", "reword", "suggest"]):
        high_or_med = [a for a in analyses if a.get("risk_level") in ("High", "Medium")]
        if not high_or_med:
            return "Everything looks low risk; no urgent rewrites needed."
        lines = ["Try these simpler, safer wordings:"]
        for x in high_or_med[:3]:
            lines.append(f"- Clause {x['clause_id']}: {x['suggested_rewrite']}")
        return "\n".join(lines)

    # Exports/help
    if any(k in text for k in ["export", "download", "json", "pdf"]):
        return "Use the Export buttons to download the JSON and PDF."

    # Default
    return (
        "Here’s how I can help: explain the overall risk, point out risky clauses, and suggest safer wording. "
        "Try: ‘What’s the overall score?’ or ‘Explain clause 3 in simple English.’"
    )


def _normalize_whitespace(text: str) -> str:
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fix_common_typos(text: str) -> str:
    # Minimal, targeted corrections for common OCR/typos seen in contracts
    replacements = {
        "transfered": "transferred",
        "tranfered": "transferred",
        "benificiar": "beneficiar",
        "beneficaries": "beneficiaries",
        "beneficary": "beneficiary",
        "liqidity": "liquidity",
        "liqiudity": "liquidity",
        "tegy": "strategy",
        "recieve": "receive",
        "recieved": "received",
        "adress": "address",
        "ammount": "amount",
        "amout": "amount",
    }
    for a, b in replacements.items():
        text = text.replace(a, b).replace(a.capitalize(), b.capitalize())
    return text


def _clean_timeline_snippet(snippet: str) -> str:
    # Clean up partials and normalize capitalization
    s = _normalize_whitespace(snippet)
    s = _fix_common_typos(s)
    # If snippet starts mid-word or with punctuation, trim to first word boundary
    import re
    m = re.search(r"[A-Za-z0-9]", s)
    if m:
        s = s[m.start():]
    # Capitalize first letter for readability
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    # Ensure sentence-like ending if missing
    if s and s[-1] not in ".!?":
        s = s + "."
    return s


def extract_timelines(text: str) -> List[Tuple[str, str]]:
    # Robust timeline extraction for PDFs using window-based snippets
    import re

    # Absolute date patterns
    month_names = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    absolute_patterns = [
        rf"\b(?:{month_names})\s+\d{{1,2}},?\s+\d{{4}}\b",              # January 5, 2024
        rf"\b\d{{1,2}}\s+(?:{month_names})\s+\d{{4}}\b",               # 05 January 2024
        rf"\b(?:{month_names})\s+\d{{4}}\b",                            # January 2024
        r"\b\d{4}-\d{2}-\d{2}\b",                                      # 2024-01-05
        r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",                   # 05/01/2024 or 05-01-24
    ]

    # Relative deadline patterns (expanded)
    rel_units = r"business\s+days?|days?|weeks?|months?|years?"
    relative_patterns = [
        rf"\bwithin\s+\d+\s+{rel_units}\b",
        rf"\bno\s+later\s+than\s+\d+\s+{rel_units}\b",
        rf"\bat\s+least\s+\d+\s+{rel_units}\s+(?:prior|before)\b",
        rf"\bnot\s+less\s+than\s+\d+\s+{rel_units}\b",
        rf"\b\d+\s+{rel_units}\s+from\s+the\s+(?:effective|execution|invoice|delivery)\s+date\b",
        rf"\bon\s+or\s+before\s+\d{{1,2}}[\/\-]\d{{1,2}}[\/\-]\d{{2,4}}\b",
        rf"\bon\s+or\s+before\s+(?:{month_names})\s+\d{{1,2}},?\s+\d{{4}}\b",
        r"\beffective\s+date\b",
        r"\btermination\s+date\b",
        r"\bstart(?:ing)?\s+on\b",
        r"\bend(?:ing)?\s+on\b",
        r"\bterm\s+of\s+\d+\s+(?:months?|years?)\b",
    ]

    # Combine all patterns with OR
    combined = re.compile("|".join([*absolute_patterns, *relative_patterns]), re.IGNORECASE)

    # Scan entire text with a sliding window because PDF sentence boundaries are unreliable
    window = 200
    hits: List[Tuple[str, str]] = []
    for m in combined.finditer(text):
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        snippet_raw = text[start:end]
        matched = m.group(0)
        snippet = _clean_timeline_snippet(snippet_raw)
        hits.append((matched, snippet))

    # Deduplicate by normalized pair
    seen = set()
    unique: List[Tuple[str, str]] = []
    for d, s in hits:
        key = (d.lower(), s)
        if key not in seen:
            seen.add(key)
            unique.append((d, s))
    return unique[:20]


def generate_document_summary(analyses: List[Dict[str, Any]], counts: Dict[str, int], overall: int) -> str:
    total = len(analyses)
    high = counts.get("High", 0)
    med = counts.get("Medium", 0)
    low = counts.get("Low", 0)
    lines = [
        f"Overall risk score: {overall}/100 across {total} clauses.",
        f"Breakdown: High {high}, Medium {med}, Low {low}.",
    ]
    if high > 0:
        top = [a for a in analyses if a.get("risk_level") == "High"][:3]
        for x in top:
            lines.append(f"- High priority: Clause {x['clause_id']} — {x['explanation']}")
    else:
        lines.append("No High‑risk clauses found. Review Medium‑risk items for small fixes.")
    lines.append("Plain English: This agreement sets who does what, by when, and what happens if things go wrong. Clearer language and liability caps usually make it safer.")
    return "\n".join(lines)


def generate_plain_english_summary(clean_text: str, max_sentences: int = 6) -> str:
    """Lightweight extractive summary in plain English style: pick informative sentences, preserve order."""
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", clean_text or "") if len(s.strip()) > 30]
    if not sentences:
        return "This document describes responsibilities, timelines, payments, and termination terms in simple language."
    # Score by length and presence of common contract terms
    keywords = {
        "term", "termination", "payment", "deliver", "obligation", "liability", "indemn", "confidential", "governing law", "notice", "timeline", "date", "days"
    }
    def score(sent: str) -> int:
        s = sent.lower()
        k = sum(1 for kw in keywords if kw in s)
        return k * 10 + min(len(sent) // 40, 5)
    ranked = sorted([(score(s), i, s) for i, s in enumerate(sentences)], reverse=True)
    chosen = sorted(ranked[:max_sentences], key=lambda x: x[1])
    out = " ".join(s for _, _, s in chosen)
    return out


if uploaded is not None:
    name = uploaded.name
    data = uploaded.read()

    text, ftype = extract_text(name, data)
    if not text:
        st.error("Could not extract any text from the uploaded file.")
        st.stop()

    pre = preprocess_contract(text)
    clean_text = pre.get("clean_text", "")
    clauses = pre.get("clauses", [])

    with st.spinner("Analyzing clauses with mock LLM..."):
        analyses = mock_llm_analyze_clauses(clauses)

    overall = compute_overall_risk_score(analyses)
    counts = summarize_risk_counts(analyses)

    result: Dict[str, Any] = {
        "file_name": name,
        "file_type": ftype,
        "overall_risk_score": overall,
        "risk_counts": counts,
        "clauses": analyses,
    }

    # Reset chat if new file
    reset_chat_for_file(file_key=f"{name}:{len(analyses)}:{overall}")

    # Sidebar: Clause-wise risk list with per-clause scores
    with st.sidebar:
        st.header("Clause-wise Risk")
        st.caption("Per-clause risk scores (High=100, Medium=50, Low=0)")
        use_claude = st.toggle("Use Claude for chat", value=True, help="Enable Anthropic Claude for chat replies.")
        for item in analyses:
            score = RISK_TO_SCORE.get(item.get("risk_level", "Low"), 0)
            label = f"Clause {item['clause_id']} — {item['risk_level']}"
            st.write(label)
            st.progress(score / 100.0, text=f"Score: {score}")
        st.divider()
        st.subheader("Exports")
        json_str = to_json(result)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="contract_analysis.json",
            mime="application/json",
            key="sidebar-json",
        )
        pdf_bytes = to_pdf_report(result)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="contract_analysis_report.pdf",
            mime="application/pdf",
            key="sidebar-pdf",
        )

    # Main content containers
    top = st.container()
    chat_area = st.container()

    with top:
        st.subheader("Plain-English Summary (Whole Document)")
        st.markdown(generate_plain_english_summary(clean_text))

        st.subheader("Document Summary")
        st.markdown(generate_document_summary(analyses, counts, overall))

        st.subheader("Important Timelines")
        timeline_hits = extract_timelines(clean_text)
        if timeline_hits:
            for date_str, sentence in timeline_hits:
                st.markdown(f"- **{date_str}**: {sentence}")
        else:
            st.info("No timelines detected. If your PDF has unusual formatting, try uploading a TXT or DOCX version.")

        st.subheader("Risk Breakdown (Chart)")
        labels = ["High", "Medium", "Low"]
        values = [counts.get(k, 0) for k in labels]
        fig = px.pie(values=values, names=labels, title="Clause Risk Levels")
        st.plotly_chart(fig, use_container_width=True)

    with chat_area:
        st.subheader("Chat with the Analyst")
        # Render chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        prompt = st.chat_input("Ask in simple English — e.g., ‘Explain clause 2’ or ‘What’s the overall risk?’")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            # Determine intent: general explanation vs. specific
            lower_q = prompt.lower()
            is_general = any(k in lower_q for k in ["explain", "about this pdf", "about this document", "summarize", "summary", "overview", "what is this pdf", "explain this pdf", "explain document"]) and not any(k in lower_q for k in ["clause "]) 

            # Build compact context: plain summary + risk + a few timelines + truncated body for grounding
            try:
                top_high = [a for a in analyses if a.get("risk_level") == "High"][:3]
                context_lines: List[str] = []
                context_lines.append("Plain-English Summary:")
                context_lines.append(generate_plain_english_summary(clean_text))
                context_lines.append("")
                context_lines.append("Risk Overview:")
                context_lines.append(generate_document_summary(analyses, counts, overall))
                if timeline_hits:
                    context_lines.append("")
                    context_lines.append("Timelines:")
                    for d, s in timeline_hits[:5]:
                        context_lines.append(f"- {d}: {s}")
                if top_high:
                    context_lines.append("")
                    context_lines.append("High-risk clauses (IDs only):")
                    context_lines.append(", ".join(str(x["clause_id"]) for x in top_high))
                if clean_text:
                    context_lines.append("")
                    context_lines.append("Document (truncated):")
                    context_lines.append(clean_text[:4000])
                context_text = "\n".join(context_lines)
            except Exception:
                context_text = clean_text[:2000]

            reply: str | None = None
            if is_general:
                # Provide local whole-document answer directly
                bullets: List[str] = []
                bullets.append(generate_plain_english_summary(clean_text))
                bullets.append("")
                bullets.append(generate_document_summary(analyses, counts, overall))
                if timeline_hits:
                    bullets.append("")
                    bullets.append("Key timelines:")
                    for d, s in timeline_hits[:5]:
                        bullets.append(f"- {d}: {s}")
                reply = "\n".join(bullets)
            else:
                if 'use_claude' in locals() and use_claude:
                    sys_prompt = (
                        "You are a legal assistant for contract review. Answer in short, plain English. "
                        "When asked to explain the document, give a concise overall explanation first and avoid focusing on a random single clause. "
                        "Only mention a clause number if the user asks for it. If unsure, say you're not sure."
                    )
                    reply = chat_with_claude(prompt, context=context_text, system_prompt=sys_prompt)
            if not reply:
                reply = generate_reply(prompt, analyses, counts, overall)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
else:
    st.info("Awaiting file upload...")
