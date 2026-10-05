import math


def normalize(images):
    """Normalize each image using its own population mean and variance."""
    normalized = []
    for image in images:
        if not image:
            raise ValueError("Cannot normalize an empty image")
        mean = sum(image) / len(image)
        variance = sum((value - mean) ** 2 for value in image) / len(image)
        scale = math.sqrt(variance + 1e-5)
        normalized.append([(value - mean) / scale for value in image])
    return normalized
