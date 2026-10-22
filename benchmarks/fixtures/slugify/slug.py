"""URL slug helper."""

import re


def slugify(value: str) -> str:
    """Return a lowercase ASCII-word slug separated by single hyphens."""

    # Deliberate fixture bug: whitespace and punctuation can produce repeated
    # or leading separators.
    return re.sub(r"[^a-z0-9 ]", "-", value.lower()).replace(" ", "-")

