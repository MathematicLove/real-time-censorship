import torch
import onnxruntime
from app import config

def available_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.insert(0, "mps")
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    return devices

def pick_device():
    wanted = str(config.FORCE_DEVICE).lower()
    devices = available_devices()
    if wanted in devices:
        return wanted
    return devices[0]

def onnx_providers(device):
    available = onnxruntime.get_available_providers()
    chain = []
    if device == "cuda":
        for name in ["TensorrtExecutionProvider", "CUDAExecutionProvider"]:
            if name in available:
                chain.append(name)
    elif device == "mps":
        for name in ["CoreMLExecutionProvider"]:
            if name in available:
                chain.append(name)
    chain.append("CPUExecutionProvider")
    return chain

def describe(device, providers):
    return "device " + device + " providers " + ",".join(providers)