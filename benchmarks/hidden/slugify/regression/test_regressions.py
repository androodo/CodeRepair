from slug import slugify


def test_already_slugged_value_is_stable() -> None:
    assert slugify("already-slugged") == "already-slugged"


def test_empty_string() -> None:
    assert slugify("") == ""

