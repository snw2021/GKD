import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


# CIFAR-100 for GKD
class CIFAR100InstanceAugment(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100InstanceAugment, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.base_transform, self.augmented_transform = transform
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # 应用基础变换（用于教师模型）
        base_img = self.base_transform(img)
        # 应用包含数据增强的变换（用于学生模型）
        augmented_img = self.augmented_transform(img)

        return base_img, augmented_img, target, index


# CIFAR-100 for CRD
class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        k=4096,
        mode="exact",
        is_sample=True,
        percent=1.0,
    ):
        super().__init__(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [
            np.asarray(self.cls_positive[i]) for i in range(num_classes)
        ]
        self.cls_negative = [
            np.asarray(self.cls_negative[i]) for i in range(num_classes)
        ]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [
                np.random.permutation(self.cls_negative[i])[0:n]
                for i in range(num_classes)
            ]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == "exact":
                pos_idx = index
            elif self.mode == "relax":
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=replace
            )
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar100_train_transform():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(p=0.1),            # 新加
            # transforms.RandomRotation(15),      # 新加
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    return train_transform


def get_cifar100_test_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

def get_cifar100_dataloaders(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100Instance(
        root=data_folder, download=True, train=True, transform=train_transform
    )
    num_data = len(train_set)
    test_set = datasets.CIFAR100(
        root=data_folder, download=True, train=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, test_loader, num_data

def get_cifar100_dataloaders_augment(batch_size, val_batch_size, num_workers):
    """增强的学生loader"""
    # 路径
    data_folder = get_data_folder()
    # base_transform, augmented_transform, test_transform
    base_transform = get_cifar100_train_transform()
    augmented_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomRotation(15),   # 新加
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = get_cifar100_test_transform()

    # train_set使用自定义的变换, test_set
    train_set = CIFAR100InstanceAugment(
        root=data_folder, download=True, train=True, transform=(base_transform, augmented_transform)
    )
    num_data = len(train_set)
    test_set = datasets.CIFAR100(
        root=data_folder, download=True, train=False, transform=test_transform
    )

    # train_loader, test_loader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, num_data


# CIFAR-100 for CRD
def get_cifar100_dataloaders_sample(
    batch_size, val_batch_size, num_workers, k, mode="exact"
):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()

    train_set = CIFAR100InstanceSample(
        root=data_folder,
        download=True,
        train=True,
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=True,
        percent=1.0,
    )
    num_data = len(train_set)
    test_set = datasets.CIFAR100(
        root=data_folder, download=True, train=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader, num_data
