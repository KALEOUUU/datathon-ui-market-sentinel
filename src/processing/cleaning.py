import json
import re
from bs4 import BeautifulSoup


DATELINE_LOCATION_PATTERN = r"^\s*(?P<location>[A-Za-z][A-Za-z .'-]{1,40}),\s*"
# Variants are constrained to the three known publishers and only accepted at
# the beginning of a source row. This handles observed source typos such as
# ``KOMPAS,com``/``KOMPAS.con`` without treating arbitrary prose as metadata.
DATELINE_PUBLISHER_PATTERNS = {
    "kompas": r"(?P<publisher>kompas(?:[\s.,]*(?:co[mn]|co[\s.]*id))?\d*)",
    "tempo": r"(?P<publisher>tempo(?:[\s.,]*(?:co[\s.]*id|co[mn]|co))?)",
    "detik": r"(?P<publisher>detik(?:[\s.,]*(?:co[mn]|co[\s.]*id))?)",
}
DATELINE_PATTERNS = {
    source: re.compile(
        DATELINE_LOCATION_PATTERN + publisher + r"\s*[–—-]{1,2}\s*",
        flags=re.IGNORECASE,
    )
    for source, publisher in DATELINE_PUBLISHER_PATTERNS.items()
}
PUBLISHER_LOCATION_DATELINE_PATTERNS = {
    source: re.compile(
        r"^\s*" + publisher + r"\s*,\s*(?P<location>[A-Za-z][A-Za-z .'-]{1,40})\s*[–—-]{1,2}\s*",
        flags=re.IGNORECASE,
    )
    for source, publisher in DATELINE_PUBLISHER_PATTERNS.items()
}
# Only observed Tempo metadata labels are removed. A generic all-caps prefix
# corrupts valid text such as "TIMNAS U-17" or "GRUP K-Pop".
TEMPO_SECTION_LABELS = frozenset(
    {
        "INFO NASIONAL",
        "INFO BISNIS",
        "MEMO BISNIS",
        "INFO TEMPO",
        "INFO OTOMOTIF",
        "INFO GAYA",
        "INFO JABAR",
    }
)
SECTION_LABEL_PATTERN = re.compile(r"^\s*(?P<label>[A-Z][A-Z ]{2,40})\s*[–—-]\s*")
QUOTE_PATTERN = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
EXPLICIT_ALIAS_PATTERN = re.compile(
    r"(?P<canonical>[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,5})\s+"
    r"\((?P<acronym>[A-Z]{2,10})\)\s+alias\s+(?P<alias>Mbak\s*[A-Z][A-Za-z'-]*|[A-Z][A-Za-z'-]*)"
)

def clean_html(text):
    """
    Remove HTML tags and residual XML markup.
    """
    if not text:
        return ""
    text = str(text)
    if "<" not in text or ">" not in text:
        return text
    # Parse HTML
    soup = BeautifulSoup(text, "lxml")
    text_clean = soup.get_text(separator=" ")
    return text_clean

def remove_boilerplate(text, source=None):
    """
    Remove common Indonesian news media boilerplates like 'Baca juga:', 'Simak video:', etc.
    """
    if not text:
        return ""

    # Define common Indonesian news boilerplate patterns
    patterns = [
        # `clean_html` intentionally flattens markup to spaces. Therefore these
        # patterns must be sentence-bounded: `[^^\n]+` would erase the remainder
        # of an article after the first related-content marker.
        r"(?i)\b(?:baca\s+juga|simak\s+video|lihat\s+foto|tonton\s+juga|pilihan\s+editor)\s*:\s*[^.?!\n]*(?:[.?!]|$)",
        r"\[Gambas:[^\]]+\]",                            # [Gambas: Video/Image]
        r"(?i)\(detik\.(?:com|co\.id)\)",                # (detik.com)/(detik.co.id)
        r"(?i)\(antara(?:\s+foto)?\)",                    # (ANTARA)/(ANTARA Foto)
        r"(?i)\bsumber\s*:\s*[^.?!\n]*(?:[.?!]|$)",      # Sumber: ...
        r"(?i)\bfoto\s*:\s*[^.?!\n]*(?:[.?!]|$)"         # Foto: ...
    ]
    if str(source or "").lower() == "tempo":
        # Repeated promotional/navigation text found in the local Tempo corpus.
        # It carries no article semantics and must not influence sentiment or TF-IDF.
        patterns.extend(
            [
                r"(?i)\bbaca\s+berita\s+dengan\s+sedikit\s+iklan,?\s+klik\s+di\s+sini\b",
                r"(?i)\bscroll\s+ke\s+bawah\s+untuk\s+melanjutkan\s+membaca\b",
            ]
        )

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned)

    # Remove URL patterns
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove email addresses
    cleaned = re.sub(r"\S+@\S+\.\S+", " ", cleaned)

    return cleaned

def normalize_whitespace(text):
    """
    Remove extra spaces, tabs, and newlines.
    """
    if not text:
        return ""
    return " ".join(text.split()).strip()

def clean_text_pipeline(text, source=None):
    """
    Run the entire cleaning pipeline for historical dataset texts.
    """
    text = clean_html(text)
    text = remove_boilerplate(text, source=source)
    text = normalize_whitespace(text)
    return text


def _normalise_dateline_location(location):
    """Correct only the known truncated dateline typo, never arbitrary content."""
    normalized = normalize_whitespace(location).upper()
    return "JAKARTA" if normalized == "AKARTA" else normalized


def _normalise_dateline_publisher(publisher, source):
    """Store a stable publisher value after matching a known source variant."""
    canonical_publishers = {"kompas": "kompas.com", "tempo": "tempo.co", "detik": "detik.com"}
    source_key = str(source).lower()
    return canonical_publishers.get(source_key, normalize_whitespace(publisher).lower())


def extract_entity_aliases(text):
    """Normalize aliases explicitly defined inside the same article.

    This is deliberately conservative: an acronym is expanded only when its
    canonical form is written in that article. Unknown acronyms remain intact.
    """
    aliases = {}

    def explicit_replacement(match):
        canonical = normalize_whitespace(match.group("canonical"))
        aliases[match.group("acronym")] = canonical
        aliases[normalize_whitespace(match.group("alias"))] = canonical
        return canonical

    normalized = EXPLICIT_ALIAS_PATTERN.sub(explicit_replacement, text)
    for alias, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", canonical, normalized)
    return normalize_whitespace(normalized), aliases


def extract_article_structure(text, source, precleaned=False):
    """Return source-aware, analysis-safe article fields without destroying meaning.

    ``content_clean`` retains readable semantic text and direct quotations.
    Quote-free and lexical forms are separate derived fields, so later model
    choices can use the appropriate representation rather than a one-size-fits-
    all destructive normalization.
    """
    semantic_text = normalize_whitespace(text) if precleaned else clean_text_pipeline(text, source=source)
    dateline_location = None
    dateline_publisher = None
    section_label = None

    # Some source rows contain the dateline twice after HTML flattening. Remove
    # every consecutive leading dateline while retaining only one metadata value.
    source_key = str(source).lower()
    dateline_patterns = (
        DATELINE_PATTERNS.get(source_key),
        PUBLISHER_LOCATION_DATELINE_PATTERNS.get(source_key),
    )
    while True:
        dateline_match = next(
            (match for pattern in dateline_patterns if pattern and (match := pattern.match(semantic_text))),
            None,
        )
        if dateline_match is None:
            break
        dateline_location = dateline_location or _normalise_dateline_location(dateline_match.group("location"))
        dateline_publisher = dateline_publisher or _normalise_dateline_publisher(dateline_match.group("publisher"), source)
        semantic_text = semantic_text[dateline_match.end():]

    section_match = SECTION_LABEL_PATTERN.match(semantic_text)
    candidate_section = normalize_whitespace(section_match.group("label")) if section_match else None
    if candidate_section in TEMPO_SECTION_LABELS and str(source).lower() == "tempo":
        section_label = candidate_section
        semantic_text = semantic_text[section_match.end():]
    semantic_text = normalize_whitespace(semantic_text)

    quote_segments = [first or second for first, second in QUOTE_PATTERN.findall(semantic_text)]
    entity_normalized_text, entity_aliases = extract_entity_aliases(semantic_text)

    return {
        "content_clean": semantic_text,
        "dateline_location": dateline_location,
        "dateline_publisher": dateline_publisher,
        "section_label": section_label,
        "quoted_text": "\n".join(quote_segments),
        "quote_count": len(quote_segments),
        "content_entity_normalized": entity_normalized_text,
        "entity_aliases_json": json.dumps(entity_aliases, ensure_ascii=False, sort_keys=True),
    }
