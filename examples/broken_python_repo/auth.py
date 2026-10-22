"""Tiny authentication policy used by the CodeRepair demonstration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRecord:
    token: str
    expires_at: int
    disabled: bool = False


def authenticate(record: TokenRecord, supplied_token: str, now: int) -> bool:
    """Return whether the supplied token is usable at ``now``."""

    if record.disabled or record.token != supplied_token:
        return False
    # Deliberate fixture bug: this comparison accepts expired tokens and rejects
    # tokens whose expiration is still in the future.
    return now >= record.expires_at

