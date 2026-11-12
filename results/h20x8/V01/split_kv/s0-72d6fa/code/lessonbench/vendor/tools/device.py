"""统一的设备选择与同步。所有实验都通过这里拿 device，保证 Mac 和 GPU 机器上代码不用改。"""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """GPU 是异步执行的：Python 发出指令后立刻返回。计时前后必须同步，否则测到的只是"发指令"的时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"cuda: {props.name}, {props.total_memory / 1e9:.0f} GB, SM x{props.multi_processor_count}"
    if device.type == "mps":
        return "mps: Apple Silicon GPU（统一内存）"
    return "cpu"


if __name__ == "__main__":
    d = get_device()
    print(describe(d))
