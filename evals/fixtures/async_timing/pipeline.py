def measure(step, synchronize, clock, steps=10, warmup=2):
    """Measure a window of asynchronous training steps in seconds."""
    for _ in range(warmup):
        step()
    started = clock()
    for _ in range(steps):
        step()
    return clock() - started
