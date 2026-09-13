"""Closed instrument and timbre vocabularies used by the annotation interface.

Annotators pick only from these. Caption pairs are generated from the schema,
so the original and swapped captions contain the same words by construction.
"""

# Pitched instruments eligible for timbre/role binding. Each entry: canonical
# name -> surface aliases used to spot the instrument in free-text captions.
INSTRUMENTS = {
    "vocals": ["vocals", "vocal", "voice", "singing", "singer", "vocalist"],
    "electric guitar": ["electric guitar", "distorted guitar", "overdriven guitar", "clean guitar"],
    "acoustic guitar": ["acoustic guitar", "nylon guitar", "classical guitar"],
    "guitar": ["guitar", "guitars"],  # generic, only if no specific guitar matched
    "piano": ["piano", "grand piano"],
    "electric piano": ["electric piano", "rhodes", "wurlitzer"],
    "bass": ["bass guitar", "bass"],
    "synth": ["synthesizer", "synth", "synths", "synth pad", "synth lead"],
    "organ": ["organ", "hammond"],
    "strings": ["strings", "string section"],
    "violin": ["violin", "fiddle"],
    "cello": ["cello"],
    "saxophone": ["saxophone", "sax"],
    "trumpet": ["trumpet"],
    "flute": ["flute"],
    "harmonica": ["harmonica"],
    # not pitched — invalid carrier for Axis T (timbre) / R (melody), but valid
    # for Axis O (onset order) and as a present instrument.
    "drums": ["drums", "drum kit", "drum set", "drumkit"],
    "percussion": ["percussion", "congas", "bongos", "tabla", "shaker"],
}

# Suggested timbre adjectives offered in the annotation interface (property T); annotators could add their own.
TIMBRES = [
    "distorted", "clean", "bright", "dark", "muted", "warm",
    "crisp", "mellow", "harsh", "soft", "punchy", "smooth",
]

# Generic terms that should not count as a distinct instrument on their own.
GENERIC = {"guitar"}


def find_instruments(caption: str):
    """Return set of canonical instruments named in a caption (specific wins
    over generic 'guitar')."""
    c = " " + caption.lower() + " "
    found = set()
    for canon, aliases in INSTRUMENTS.items():
        for a in aliases:
            if a in c:
                found.add(canon)
                break
    # specific beats generic
    if found & {"acoustic guitar", "electric guitar"} and "guitar" in found:
        found.discard("guitar")
    if "electric piano" in found and "piano" in found:
        found.discard("piano")
    if "double bass" in found and "bass" in found:
        found.discard("bass")
    return found
