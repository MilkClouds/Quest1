from torchvision.datasets import Flowers102
from torchvision import transforms
import albumentations as A
image_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
def get_transforms(image_size=224):
    transform_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
            A.Resize(image_size, image_size),
            A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    transform_valid = A.Compose([
            A.Resize(image_size, image_size),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform_train, transform_valid
root = '/mnt/d/datasets'
def load_dataset(mode): # pretrain / finetune
    transforms_train, transforms_valid = get_transforms()
    train_dataset = Flowers102(root=root, split="train", transform=image_preprocess, download=True)
    validation_dataset = Flowers102(root=root, split="val", transform=image_preprocess)
    test_dataset = Flowers102(root=root, split="test", transform=image_preprocess)
    return train_dataset, validation_dataset, test_dataset