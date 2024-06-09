import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
from datasets import load_dataset


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

    def __len__(self):
        return len(self.dataset)


def get_dataloader(dataset_name="pcuenq/lsun-bedrooms", image_size=64, batch_size=128):
    train_dataset = TorchDataset(
        load_dataset(dataset_name)["train"],
        transform=transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return dataloader
