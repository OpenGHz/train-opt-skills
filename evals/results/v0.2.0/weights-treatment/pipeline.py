from numbers import Integral


def convert_qkv(weight, num_heads):
    """Copy head-interleaved Q/K/V rows into Q-, K-, then V-major order."""
    if isinstance(num_heads, bool) or not isinstance(num_heads, Integral) or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    if not isinstance(weight, list) or len(weight) != 3 * num_heads:
        raise ValueError("weight must be a list with 3 * num_heads rows")
    if any(not isinstance(row, list) or not row for row in weight):
        raise ValueError("weight rows must be nonempty lists")
    width = len(weight[0])
    if any(len(row) != width for row in weight):
        raise ValueError("weight rows must have equal width")

    return [
        weight[3 * head + component][:]
        for component in range(3)
        for head in range(num_heads)
    ]
