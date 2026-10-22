import pytest
from chunks import chunks


def test_empty_input() -> None:
    assert chunks([], 3) == []


@pytest.mark.parametrize("size", [0, -1])
def test_bad_size(size: int) -> None:
    with pytest.raises(ValueError):
        chunks([1], size)

