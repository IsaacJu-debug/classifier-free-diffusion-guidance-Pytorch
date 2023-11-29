from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

def load_data(batchsize:int, numworkers:int) -> DataLoader:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root = './dataset',
                        train = True,
                        download = False,
                        transform = trans
                    )
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        shuffle = True,
                        num_workers = numworkers,
                        drop_last = True
                    )
    return trainloader

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
