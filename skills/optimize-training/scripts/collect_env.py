#!/usr/bin/env python3
"""Collect an allowlisted environment summary without dumping environment variables.

Default: Python/platform and installed package versions via importlib.metadata;
no deep-learning framework imports. --probe-torch runs an explicit subprocess
probe (20s timeout, may initialize CUDA). --nvidia-smi calls the NVIDIA CLI (5s
 timeout). Only GPU index, name, driver and total memory are requested; no UUIDs,
process lists, hostnames, paths or environment variable values are collected.
Review output before sharing: hardware/software versions may still be sensitive.
Exit: 0 report created (optional probe failures recorded); 2 output error.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import subprocess
import sys
from _common import emit

PACKAGES = ("torch", "torchvision", "torchaudio", "numpy", "transformers", "accelerate",
            "deepspeed", "jax", "jaxlib", "triton", "flash-attn", "datasets", "safetensors")


def probe(command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode:
            # Child stderr may contain local paths or other incidental details.
            return {"status":"failed", "returncode":result.returncode}
        return {"status":"ok", "stdout":result.stdout[:16384]}
    except FileNotFoundError:
        return {"status":"unavailable"}
    except subprocess.TimeoutExpired:
        return {"status":"timeout", "timeout_seconds":timeout}
    except OSError:
        return {"status":"failed_to_execute"}


def collect(probe_torch=False, nvidia_smi=False):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"schema_version":1, "collected_at_utc":datetime.now(timezone.utc).isoformat(),
              "python":{"version":platform.python_version(),"implementation":platform.python_implementation()},
              "platform":{"system":platform.system(), "release":platform.release(), "machine":platform.machine()},
              "packages":versions, "environment_variables_collected":False}
    if nvidia_smi:
        queried = probe(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"], 5)
        if queried["status"] == "ok":
            queried["csv_columns"] = ["index", "name", "driver_version", "memory_total_mib"]
        result["nvidia_smi"] = queried
    if probe_torch:
        code = ('import json,torch; print(json.dumps({'
                '"torch_version":str(torch.__version__),"cuda_build_version":torch.version.cuda,'
                '"cuda_available":torch.cuda.is_available(),"cuda_device_count":torch.cuda.device_count(),'
                '"cudnn_version":torch.backends.cudnn.version()}))')
        queried = probe([sys.executable, "-c", code], 20)
        if queried["status"] == "ok":
            try:
                queried["details"] = json.loads(queried.pop("stdout"))
            except (ValueError, KeyError):
                queried = {"status":"invalid_probe_output"}
        result["torch_probe"] = queried
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-torch", action="store_true", help="Explicitly import torch in a timed child process")
    parser.add_argument("--nvidia-smi", action="store_true", help="Query allowlisted GPU metadata with a 5-second timeout")
    parser.add_argument("--output", help="Write JSON report")
    args = parser.parse_args()
    try:
        emit(collect(args.probe_torch, args.nvidia_smi), args.output)
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
