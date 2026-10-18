from soznet.data.labels import parse_blocks_to_multilabel, flip_label


def test_parse_blocks_explicit():
    # 1 -> LF, 3 -> LTP, 5 -> LO, all left
    assert parse_blocks_to_multilabel("1,3,5", "L") == [1, 0, 1, 0, 1, 0]
    # 2 -> RF, 4 -> RTP, 6 -> RO, all right
    assert parse_blocks_to_multilabel("2,4,6", "R") == [0, 1, 0, 1, 0, 1]


def test_parse_blocks_empty_uses_position():
    assert parse_blocks_to_multilabel("", "L") == [1, 0, 1, 0, 1, 0]
    assert parse_blocks_to_multilabel(None, "R") == [0, 1, 0, 1, 0, 1]


def test_flip_label_swaps_sides():
    assert flip_label([1, 0, 1, 0, 0, 1]) == [0, 1, 0, 1, 1, 0]
