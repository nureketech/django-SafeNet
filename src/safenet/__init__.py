"""SafeNet KZ: detection of *signs* of possible illegal schemes in social media content.

The system never declares content illegal. It reports signals of a possible
violation, a risk level and an explanation; a human specialist decides.
"""

__version__ = "0.1.0"

DISCLAIMER_KK = (
    "Ықтимал бұзушылықтың белгілері анықталды. "
    "ҚР заңнамасын ескере отырып, маманның тексеруі қажет."
)
DISCLAIMER_RU = (
    "Выявлены признаки возможного нарушения. "
    "Требуется проверка специалистом с учётом законодательства РК."
)

TRACKS: tuple[str, ...] = ("lottery", "casino", "pyramid")
LABELS: tuple[str, ...] = (*TRACKS, "none")
