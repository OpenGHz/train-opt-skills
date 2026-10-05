import math


def normalize(images):
    flat = [value for image in images for value in image]
    mean = sum(flat) / len(flat)
    variance = sum((value - mean) ** 2 for value in flat) / len(flat)
    return [[(value - mean) / math.sqrt(variance + 1e-5) for value in image] for image in images]
