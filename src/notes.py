"""Email notes that carry a severity and a short title.

A `Note` IS a `str` (the sentence shown to the reader), so every place that treats notes as plain
strings (lists, `in` checks, `html.escape`) keeps working. The email renderer reads the two extra
attributes to group notes into "perlu diperhatikan" and "info" and to print a bold title.

Notes never contain API keys or addresses. They may mention dates and counts; the pipeline does not
write them to the (public) Actions log, only to the private email.
"""

from __future__ import annotations

WARN = "warn"      # affects how much to trust today's numbers
INFO = "info"      # context only


class Note(str):
    """The attribute is called `heading`, not `title`: `str.title` is a built-in method, so a plain
    string would answer getattr(n, "title") with that method instead of a missing-attribute default."""

    level: str
    heading: str

    def __new__(cls, heading: str, text: str, level: str = WARN) -> "Note":
        obj = super().__new__(cls, text)
        obj.level = level if level in (WARN, INFO) else WARN
        obj.heading = heading
        return obj


def note_level(n: str) -> str:
    """Severity of any note. Plain strings (older code paths, news notes) count as WARN."""
    return n.level if isinstance(n, Note) else WARN


def note_title(n: str) -> str:
    """Bold heading of a note; empty for plain strings."""
    return n.heading if isinstance(n, Note) else ""
