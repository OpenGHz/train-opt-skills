def convert_qkv(weight, num_heads):
    # Same shape, so the first version treated conversion as a copy.
    return [row[:] for row in weight]
