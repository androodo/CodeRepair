from slug import slugify


def test_collapses_separators() -> None:
    assert slugify("Hello,   Python!") == "hello-python"


def test_simple_words() -> None:
    assert slugify("Code Repair") == "code-repair"

