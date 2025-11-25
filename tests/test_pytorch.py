import torch


def main():
    # Asserting that my xpu is detected
    assert torch.xpu.is_available(), "XPU is not available."
    assert torch.xpu.device_count() > 0, "No XPU devices found."

    x = torch.randn(3, 3, device="xpu")
    y = torch.randn(3, 3, device="xpu")
    z = torch.matmul(x, y)
    print(z)
    print(f"Resulting tensor is on device: {z.device}")
    print(f"GPU: {torch.xpu.get_device_name(0)}")


if __name__ == "__main__":
    main()
