"""Text normalization for business names and addresses.

All normalization is applied identically to every source, so noisy variants
(abbreviations, legal suffixes, accents, state names) collapse to a common form.
No external data or APIs are used.
"""
import re
import unicodedata

# ---------------------------------------------------------------- basic text

_WS = re.compile(r"\s+")
_NOT_WORD = re.compile(r"[^\w\s]+", re.UNICODE)  # keeps unicode letters (Devanagari etc.)


def _latin1_fold_table():
    """Map accented Latin-1 chars to their ASCII base (translate table).

    Only touches U+00C0..U+00FF, so Indic scripts are never corrupted.
    """
    table = {}
    for c in range(0xC0, 0x100):
        base = unicodedata.normalize("NFD", chr(c))
        base = "".join(ch for ch in base if not unicodedata.combining(ch))
        if len(base) == 1 and ord(base) < 128 and base.isprintable():
            table[chr(c)] = base
    table.update({"æ": "ae", "œ": "oe", "ø": "o", "ð": "d", "þ": "th"})
    return str.maketrans(table)


_LATIN1_FOLD = _latin1_fold_table()


def norm_text(s):
    """Casefold, fold latin accents, '&'->'and', strip punctuation, collapse ws."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = s.translate(_LATIN1_FOLD)
    if "&" in s:
        s = s.replace("&", " and ")
    s = _NOT_WORD.sub(" ", s)
    s = s.replace("_", " ")
    return _WS.sub(" ", s).strip()


# ------------------------------------------------------------- legal suffixes

LEGAL_TOKENS = frozenset({
    # English
    "ltd", "limited", "llc", "llp", "lp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "pvt", "pvtl", "private", "plc",
    "and",  # '&' was expanded to 'and'; dropping merges "A & B" with "A B"
    # European
    "gmbh", "kg", "ag", "ohg", "sarl", "sas", "eurl", "srl", "spa", "sa",
    "bv", "nv", "oy", "ab", "as", "oyj", "pty", "pte", "ltda", "eirl",
    "ets", "ste", "sci",
    # transliterated Indian legal suffixes
    "प्राइवेट", "लिमिटेड", "प्रा", "लि", "प्रा।", "लि।",
    "ലിമിറ്റഡ്", "പ്രൈവറ്റ്", "പ്രാ",
    "லிமிடெட்", "பிரைவேட்",
    "లిమిటెడ్", "ప్రైవేట్",
    "ಲಿಮಿಟೆಡ್", "ಪ್ರೈವೇಟ್",
    "লিমিটেড", "প্রাইভেট",
    "લિમિટેડ", "પ્રાઇવેટ",
    "ਲਿਮਿਟੇਡ", "ਪ੍ਰਾਈਵੇਟ",
})


def name_tokens(norm_name):
    """Tokens of a normalized name with legal suffixes dropped."""
    if not norm_name:
        return []
    return [t for t in norm_name.split() if t not in LEGAL_TOKENS]


def name_sig(norm_name):
    """Order-insensitive signature of a name (legal suffixes dropped)."""
    toks = name_tokens(norm_name)
    toks.sort()
    return " ".join(toks)


# ------------------------------------------------------------------ addresses

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
_STATES_RE = re.compile(
    r"\b(?:" + "|".join(sorted((re.escape(k) for k in US_STATES), key=len, reverse=True)) + r")\b"
)


def _states_sub(m):
    return US_STATES[m.group(0)]


# street-type / unit abbreviations, applied to every token EXCEPT the first
# (first token is often a house number or a "saint" abbreviation).
_STREET = {
    "rd": "road", "road": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "avenue": "avenue", "dr": "drive", "blvd": "boulevard", "ln": "lane",
    "ct": "court", "cir": "circle", "pkwy": "parkway", "hwy": "highway",
    "ter": "terrace", "trl": "trail", "sq": "square", "expy": "expressway",
    "fwy": "freeway", "pl": "place", "plz": "plaza",
    "ste": "apt", "suite": "apt", "unit": "apt",
}


def norm_name(s):
    """Normalized business name (legal suffixes kept; use name_sig for keys)."""
    return norm_text(s)


def norm_addr(s):
    """Normalized address: punctuation folded, US states -> abbrev, street types expanded."""
    a = norm_text(s)
    if not a:
        return ""
    if "null" in a:
        a = a.replace(" null ", " ").replace("null ", " ").replace(" null", " ").strip()
        a = _WS.sub(" ", a)
    a = _STATES_RE.sub(_states_sub, a)
    toks = a.split()
    if len(toks) > 1:
        head, rest = toks[0], toks[1:]
        a = " ".join([head] + [_STREET.get(t, t) for t in rest])
    return a


# ------------------------------------------------------------ component keys

_HOUSE_RE = re.compile(r"\d{1,7}")
_ZIP6_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
_ZIP5_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")


def house_num(addr_norm):
    m = _HOUSE_RE.search(addr_norm)
    return m.group(0) if m else ""


def postal_code(addr_norm):
    m = _ZIP6_RE.search(addr_norm) or _ZIP5_RE.search(addr_norm)
    return m.group(0) if m else ""


def has_nonascii(s):
    return any(ord(c) > 127 for c in s)
