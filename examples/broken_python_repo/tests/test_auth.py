from auth import TokenRecord, authenticate


def test_future_expiration_is_valid() -> None:
    assert authenticate(TokenRecord("secret", expires_at=200), "secret", now=100)


def test_expired_token_is_invalid() -> None:
    assert not authenticate(TokenRecord("secret", expires_at=50), "secret", now=100)


def test_disabled_token_is_invalid() -> None:
    assert not authenticate(TokenRecord("secret", expires_at=200, disabled=True), "secret", 100)

