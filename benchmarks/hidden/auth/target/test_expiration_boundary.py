from auth import TokenRecord, authenticate


def test_token_is_invalid_at_exact_expiration() -> None:
    assert not authenticate(TokenRecord("secret", expires_at=100), "secret", now=100)

