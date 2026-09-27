"""
tokenizer.py

Splits code and docstrings into search terms. Generic word tokenizers treat
`getUserById` and `get_user_by_id` as unrelated single tokens; splitting
camelCase and snake_case identifiers into their parts means a query like
"user id" matches both, without needing an embedding model at all.
"""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z0-9]+")

# Deliberately tiny: this is code search, not prose search. Filtering out
# common English stopwords would also filter out real identifiers like
# `is`, `as`, `in`, `for` that show up constantly in code and signatures.
_STOPWORDS = frozenset({"the", "a", "an", "of", "to"})


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    # `_WORD` already treats `_` as a separator, so `get_user_by_id` comes
    # back as four words; `_CAMEL_BOUNDARY` then splits each of those on
    # camelCase, so `getUserById` also comes back as four.
    tokens: list[str] = []
    for word in _WORD.findall(text):
        for sub in _CAMEL_BOUNDARY.split(word):
            sub = sub.lower()
            if sub and sub not in _STOPWORDS:
                tokens.append(sub)
    return tokens
