import pytest


@pytest.mark.skip
def test_gpu_util():
    torch = pytest.importorskip("torch")

    print("torch.cuda.is_available()", torch.cuda.is_available())
    print("torch.cuda.current_device()", torch.cuda.current_device())
    print("torch.cuda.device(0)", torch.cuda.device(0))
    print("torch.cuda.device_count()", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    test_gpu_util()
