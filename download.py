from torchvision.datasets import CIFAR10
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
testset  = CIFAR10(root="./data", train=False, download=True, transform=transform)