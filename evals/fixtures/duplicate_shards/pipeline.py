def shard(data, rank, world_size, worker_id=0, num_workers=0):
    effective_workers = max(1, num_workers)
    return data[worker_id::effective_workers * world_size]
