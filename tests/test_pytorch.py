import torch


def main():
    # check which device is available
    if torch.cuda.is_available():
        device = "cuda"
        print("Available CUDA devices:", torch.cuda.device_count())
    elif torch.xpu.is_available():
        device = "xpu"
        print("Available XPU devices:", torch.xpu.device_count())
    else:
        device = "cpu"

    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = torch.matmul(x, y)
    print(z)
    print(f"Resulting tensor is on device: {z.device}")


if __name__ == "__main__":
    main()
