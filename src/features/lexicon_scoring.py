"""Deterministic, conservative scoring utilities for the local InSet lexicon."""

from __future__ import annotations

from dataclasses import dataclass
import re


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokens used by every lexicon-based baseline."""
    return TOKEN_PATTERN.findall(str(text).lower())


def _normalise_term(term: object) -> tuple[str, ...]:
    return tuple(tokenize(str(term)))


@dataclass(frozen=True)
class InSetScorer:
    """Longest-match InSet scorer with an explicit ambiguity policy.

    A lexicon entry that is present in both polarity files is omitted entirely.
    InSet alone does not disambiguate its senses, so choosing either file's
    weight would fabricate a directional signal. Remaining entries are matched
    longest-first and non-overlapping, preventing a phrase and its constituent
    words from being counted twice.
    """

    entries_by_first_token: dict[str, tuple[tuple[tuple[str, ...], float], ...]]
    overlap_terms: frozenset[tuple[str, ...]]
    positive_phrase_count: int
    negative_phrase_count: int

    def density(self, text: str) -> float:
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        score = 0.0
        index = 0
        while index < len(tokens):
            match = next(
                (
                    (term_tokens, weight)
                    for term_tokens, weight in self.entries_by_first_token.get(tokens[index], ())
                    if tokens[index : index + len(term_tokens)] == list(term_tokens)
                ),
                None,
            )
            if match is None:
                index += 1
                continue
            term_tokens, weight = match
            score += weight
            index += len(term_tokens)
        return max(-1.0, min(1.0, score / (5.0 * len(tokens))))


def build_inset_scorer(positive: dict[str, float], negative: dict[str, float]) -> InSetScorer:
    """Build a scorer from raw InSet maps without silently resolving conflicts."""
    positive_terms = {_normalise_term(term): float(weight) for term, weight in positive.items() if _normalise_term(term)}
    negative_terms = {_normalise_term(term): float(weight) for term, weight in negative.items() if _normalise_term(term)}
    overlap_terms = frozenset(positive_terms.keys() & negative_terms.keys())
    entries = {
        **{term: weight for term, weight in positive_terms.items() if term not in overlap_terms},
        **{term: weight for term, weight in negative_terms.items() if term not in overlap_terms},
    }
    by_first: dict[str, list[tuple[tuple[str, ...], float]]] = {}
    for term_tokens, weight in entries.items():
        by_first.setdefault(term_tokens[0], []).append((term_tokens, weight))
    sorted_entries = {
        first: tuple(sorted(values, key=lambda item: len(item[0]), reverse=True))
        for first, values in by_first.items()
    }
    return InSetScorer(
        entries_by_first_token=sorted_entries,
        overlap_terms=overlap_terms,
        positive_phrase_count=sum(len(term) > 1 for term in positive_terms if term not in overlap_terms),
        negative_phrase_count=sum(len(term) > 1 for term in negative_terms if term not in overlap_terms),
    )
