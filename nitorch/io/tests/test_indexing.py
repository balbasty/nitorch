from ..utils import indexing as idx


def test_neg2pos():
    assert idx.neg2pos(0, 3) == 0, "positive input"
    assert idx.neg2pos(-1, 3) == 2, "negative input -> positive outpu"
    assert idx.neg2pos(-3, 3) == 0, "negative input -> zero"
    assert idx.neg2pos(-5, 3) == -2, "negative input -> negative output"
    assert idx.neg2pos(slice(-1, None), 3) == slice(2, None), "slice"
    assert idx.neg2pos(None, 3) is None, "none"
    assert idx.neg2pos(Ellipsis, 3) is Ellipsis, "ellipsis"
    try: idx.neg2pos(0, -1)
    except ValueError: pass
    else: assert False, "raise on negative shape"
    try:  idx.neg2pos(0, 3.5)
    except TypeError: pass
    else: assert False, "raise on non integer shape"
    try:  idx.neg2pos(4., 3)
    except TypeError: pass
    else: assert False, "raise on non index_like"
    try:  idx.neg2pos((slice(None), slice(None)), [4])
    except ValueError: pass
    else: assert False, "shape too short"
    try:  idx.neg2pos((slice(None), None, slice(None)), [4, 5, 6])
    except ValueError: pass
    else: assert False, "shape too long"


def test_simplify_slice():
    # stride == +1
    assert idx.simplify_slice(slice(None), 3) == slice(None), "full: ="
    assert idx.simplify_slice(slice(3), 3) == slice(None), "full: stop ="
    assert idx.simplify_slice(slice(4), 3) == slice(None), "full: stop >"
    assert idx.simplify_slice(slice(0, None), 3) == slice(None), "full: start ="
    assert idx.simplify_slice(slice(-5, None), 3) == slice(None), "full: start < (using neg index)"
    assert idx.simplify_slice(slice(-5, 4), 3) == slice(None), "full: start < stop >"
    assert idx.simplify_slice(slice(0, 3), 3) == slice(None), "full: start+stop ="
    assert idx.simplify_slice(slice(2), 3) == slice(2), "sub: stop <"
    assert idx.simplify_slice(slice(1, None), 3) == slice(1, None), "sub: start >"
    assert idx.simplify_slice(slice(1, 2), 3) == slice(1, 2), "sub: start > stop <"
    assert idx.simplify_slice(slice(1, 4), 3) == slice(1, None), "sub: start > stop >"
    assert idx.simplify_slice(slice(-5, 2), 3) == slice(2), "sub: start < stop <"
    assert idx.simplify_slice(slice(None, None, 1), 3) == slice(None), "full: step ="
    # stride == -1
    assert idx.simplify_slice(slice(None, None, -1), 3) == slice(None, None, -1), "inv"
    assert idx.simplify_slice(slice(2, None, -1), 3) == slice(None, None, -1), "inv full: start ="
    assert idx.simplify_slice(slice(3, None, -1), 3) == slice(None, None, -1), "inv full: start <"
    assert idx.simplify_slice(slice(None, -5, -1), 3) == slice(None, None, -1), "inv full: stop > (using neg index)"
    assert idx.simplify_slice(slice(4, -5, -1), 3) == slice(None, None, -1), "inv full: start < stop >"
    assert idx.simplify_slice(slice(2, -4, -1), 3) == slice(None, None, -1), "inv full: start+stop ="
    assert idx.simplify_slice(slice(1, None, -1), 3) == slice(1, None, -1), "inv sub: start >"
    assert idx.simplify_slice(slice(None, 1, -1), 3) == slice(None, 1, -1), "inv sub: stop <"
    assert idx.simplify_slice(slice(2, 1, -1), 3) == slice(None, 1, -1), "inv sub: start > stop <"
    assert idx.simplify_slice(slice(4, 1, -1), 3) == slice(None, 1, -1), "inv sub: start < stop <"
    assert idx.simplify_slice(slice(1, -5, -1), 3) == slice(1, None, -1), "inv sub: start > stop >"
    # stride even
    assert idx.simplify_slice(slice(None, None, 2), 3) == slice(None, None, 2), "even: ="
    assert idx.simplify_slice(slice(None, 4, 2), 4) == slice(None, None, 2), "even: stop ="
    assert idx.simplify_slice(slice(None, 5, 2), 4) == slice(None, None, 2), "even: stop >"
    assert idx.simplify_slice(slice(0, None, 2), 3) == slice(None, None, 2), "even: start ="
    assert idx.simplify_slice(slice(-5, None, 2), 3) == slice(None, None, 2), "even: start < (using neg index)"
    assert idx.simplify_slice(slice(-5, 4, 2), 3) == slice(None, None, 2), "even: start < stop >"
    assert idx.simplify_slice(slice(0, 3, 2), 3) == slice(None, None, 2), "even: start+stop ="
    assert idx.simplify_slice(slice(None, 2, 2), 3) == slice(None, 2, 2), "even sub: stop <"
    assert idx.simplify_slice(slice(1, None, 2), 3) == slice(1, None, 2), "even sub: start >"
    assert idx.simplify_slice(slice(1, 3, 2), 5) == slice(1, 3, 2), "even sub: start > stop <"
    assert idx.simplify_slice(slice(1, 2, 2), 3) == slice(1, None, 2), "even sub: start > stop <="
    assert idx.simplify_slice(slice(1, 4, 2), 3) == slice(1, None, 2), "even sub: start > stop >"
    assert idx.simplify_slice(slice(-5, 2, 2), 3) == slice(None, 2, 2), "even sub: start < stop <"


def test_expand_index():
    # index expansion
    assert idx.expand_index([Ellipsis], [2, 3]) == (slice(None), slice(None)), "ellipsis"
    assert idx.expand_index([slice(None)], [2, 3]) == (slice(None), slice(None)), "implicit dims"
    assert idx.expand_index([None, 1, slice(None)], [2, 3]) == (None, 1, slice(None)), "none"
    assert idx.expand_index([1, slice(None)], [2, 3]) == (1, slice(None)), "int"
    try:  idx.expand_index([2, slice(None)], [2, 3])
    except IndexError: pass
    else: assert False, "oob index"


def test_guess_shape():
    # shape calculator
    assert idx.guess_shape([Ellipsis], [2, 3]) == (2, 3), "ellipsis"
    assert idx.guess_shape([slice(None), slice(2, None)], [2, 3]) == (2, 1), "slice"
    assert idx.guess_shape([slice(None), slice(None, 1, -1)], [2, 3]) == (2, 1), "slice inv"
    assert idx.guess_shape([slice(None), 0], [2, 3]) == (2,), "drop"
    assert idx.guess_shape([slice(None), None, slice(2, None)], [2, 3]) == (2, 1, 1), "new axis"
    assert idx.guess_shape([slice(None, None, 1)], [1]) == (1,), "::1 into 1"
    assert idx.guess_shape([slice(None, None, 2)], [1]) == (1,), "::2 into 1"
    assert idx.guess_shape([slice(None, None, 1)], [2]) == (2,), "::1 into 2"
    assert idx.guess_shape([slice(None, None, 2)], [2]) == (1,), "::2 into 2"
    assert idx.guess_shape([slice(None, None, 3)], [2]) == (1,), "::3 into 2"
    assert idx.guess_shape([slice(None, None, 1)], [3]) == (3,), "::1 into 3"
    assert idx.guess_shape([slice(None, None, 2)], [3]) == (2,), "::2 into 3"
    assert idx.guess_shape([slice(None, None, 3)], [3]) == (1,), "::3 into 3"
    assert idx.guess_shape([slice(None, None, 4)], [3]) == (1,), "::4 into 3"
    assert idx.guess_shape([slice(None, None, 1)], [4]) == (4,), "::1 into 4"
    assert idx.guess_shape([slice(None, None, 2)], [4]) == (2,), "::2 into 4"
    assert idx.guess_shape([slice(None, None, 3)], [4]) == (2,), "::3 into 4"
    assert idx.guess_shape([slice(None, None, 4)], [4]) == (1,), "::4 into 4"
    assert idx.guess_shape([slice(None, None, 5)], [4]) == (1,), "::5 into 4"


def test_compose_index():
    assert idx.compose_index([slice(None), slice(None)], [slice(None), slice(None)], [5, 5]) == (slice(None), slice(None)), "full of full"
    assert idx.compose_index([None, slice(None)], [slice(None), slice(None)], [5]) == (None, slice(None)), "slice of newaxis"
    assert idx.compose_index([0, slice(None)], [slice(None)], [5, 5]) == (0, slice(None)), "dropped axis"
    assert idx.compose_index([slice(2, None)], [slice(1, None)], [5]) == (slice(3, None),), "compose starts"
    assert idx.compose_index([slice(2, None)], [slice(-2, None, -1)], [5]) == (slice(3, 1, -1),), "compose starts inverse"
    # raised errors
    try:  idx.compose_index([None], [1], [])
    except IndexError: pass
    else: assert False, "oob: newaxis[int]"
    try:  idx.compose_index([slice(None)], [5], [3])
    except IndexError: pass
    else: assert False, "oob: slice[int]"
    try:  idx.compose_index([idx.oob_slice()], [0], [3])
    except IndexError: pass
    else: assert False, "oob: oob_slice[int]"
