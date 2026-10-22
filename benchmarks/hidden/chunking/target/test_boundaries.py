from chunks import chunks


def test_exact_multiple_keeps_last_full_chunk() -> None:
    assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_single_item() -> None:
    assert chunks(["only"], 3) == [["only"]]

