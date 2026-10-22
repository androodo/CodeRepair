from chunks import chunks


def test_keeps_partial_final_chunk() -> None:
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_rejects_nonpositive_size() -> None:
    try:
        chunks([1], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

