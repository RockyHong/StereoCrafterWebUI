"""Detect CUDA version and print the best PyTorch index tag.

Checks both the GPU driver (nvidia-smi) and the CUDA toolkit (nvcc).
Uses the LOWER of the two, since compiling CUDA extensions requires
the toolkit version to match PyTorch's CUDA version.
"""
import subprocess, re


def get_driver_cuda():
    """Get max CUDA version supported by the GPU driver."""
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def get_toolkit_cuda():
    """Get installed CUDA toolkit version from nvcc."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def version_to_tag(major, minor):
    if major == 11:
        return "cu118"
    elif major == 12 and minor <= 1:
        return "cu121"
    elif major == 12 and minor <= 4:
        return "cu124"
    elif major == 12 and minor <= 7:
        return "cu126"
    elif major == 12:
        return "cu128"
    else:
        return "cu130"


driver = get_driver_cuda()
toolkit = get_toolkit_cuda()

if driver is None and toolkit is None:
    print("cpu")
elif toolkit is not None and driver is not None:
    # Use the lower version — toolkit must match PyTorch for compiling extensions
    use = min(driver, toolkit)
    print(version_to_tag(*use))
elif toolkit is not None:
    print(version_to_tag(*toolkit))
else:
    print(version_to_tag(*driver))
