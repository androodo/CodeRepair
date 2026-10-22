from slug import slugify


def test_trims_leading_and_trailing_separators() -> None:
    assert slugify("  Hello world!!! ") == "hello-world"


def test_mixed_punctuation_is_one_separator() -> None:
    assert slugify("api_._client") == "api-client"

