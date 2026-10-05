import math


def normalize(images):
    """Normalize each nonempty image using its own population statistics."""
    normalized = []
    for image in images:
        if not image:
            raise ValueError("cannot normalize an empty image")
        mean = sum(image) / len(image)
        variance = sum((value - mean) ** 2 for value in image) / len(image)
        denominator = math.sqrt(variance + 1e-5)
        normalized.append([(value - mean) / denominator for value in image])
    return normalized
