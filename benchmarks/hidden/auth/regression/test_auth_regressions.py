from auth import TokenRecord, authenticate


def test_wrong_token_never_authenticates() -> None:
    assert not authenticate(TokenRecord("right", expires_at=200), "wrong", now=100)


def test_disabled_future_token_never_authenticates() -> None:
    record = TokenRecord("right", expires_at=200, disabled=True)
    assert not authenticate(record, "right", now=100)

