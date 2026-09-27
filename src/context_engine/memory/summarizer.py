"""
summarizer.py

Turns raw session rows (events, decisions, code areas) into a compact
string an agent can be handed at the start of a turn instead of full chat
history. Deliberately extractive and deterministic — no LLM call, so
building a brief costs zero tokens and zero latency. If CodeVeto later
wants a richer, abstractive brief, that's a second implementation behind
the same `build_brief` signature, not a change to this one.
"""

from __future__ import annotations

from models.context_engine_models import CodeArea, Decision, SessionEvent

_ELLIPSIS = "\n...(truncated to fit budget)"


def compress_events(
    events: list[SessionEvent], max_items: int = 10
) -> list[SessionEvent]:
    """Drop to the most recent `max_items`, collapsing consecutive
    duplicate (event_type) runs down to one representative entry plus a
    count, so ten identical "read_file" events don't crowd out everything
    else in the brief."""
    if not events:
        return []
    collapsed: list[SessionEvent] = []
    counts: dict[str, int] = {}
    for event in events:  # events are newest-first
        if collapsed and collapsed[-1].event_type == event.event_type:
            counts[event.event_type] = counts.get(event.event_type, 1) + 1
            continue
        collapsed.append(event)
        counts.setdefault(event.event_type, 1)
        if len(collapsed) >= max_items:
            break
    for event in collapsed:
        if counts.get(event.event_type, 1) > 1:
            event.data = {**event.data, "_repeated": counts[event.event_type]}
    return collapsed


def build_brief(
    decisions: list[Decision],
    code_areas: list[CodeArea],
    events: list[SessionEvent],
    budget_chars: int,
) -> str:
    """All three inputs are expected newest-first (that's what
    `SessionStore.recent_*` returns) so truncation naturally drops the
    oldest material first."""
    lines: list[str] = []

    if decisions:
        lines.append("Decisions so far:")
        for d in decisions:
            rationale = f" ({d.rationale})" if d.rationale else ""
            lines.append(f"- {d.decision}{rationale}")

    if code_areas:
        lines.append("Code areas flagged as relevant:")
        for a in code_areas:
            note = f" — {a.note}" if a.note else ""
            lines.append(f"- {a.file_path}{note}")

    compressed_events = compress_events(events)
    if compressed_events:
        lines.append("Recent activity:")
        for e in compressed_events:
            repeated = e.data.get("_repeated")
            suffix = f" (x{repeated})" if repeated else ""
            lines.append(f"- {e.event_type}{suffix}")

    if not lines:
        return "No prior activity recorded for this session."

    text = "\n".join(lines)
    return _truncate(text, budget_chars)


def _truncate(text: str, budget_chars: int) -> str:
    if len(text) <= budget_chars:
        return text
    cutoff = max(budget_chars - len(_ELLIPSIS), 0)
    return text[:cutoff] + _ELLIPSIS
