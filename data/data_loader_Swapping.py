import os
import glob
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class InferenceDataset(data.Dataset):

    def __init__(self, root, transform=None):
        self.paths = sorted(make_dataset(root))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB')
        if self.transform:
            from_im = self.transform(from_im)
        return from_im


class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.num_images = len(loader)
        self.preload()

    def preload(self):
        try:
            self.src_image = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.src_image = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.src_image = self.src_image.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        src_image = self.src_image
        self.preload()
        return src_image

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def GetLoader(dataset_roots,
              batch_size=16,
              dataloader_workers=8,
              random_seed=1234
              ):
    """Build and return a data loader."""

    num_workers = dataloader_workers
    data_root = dataset_roots
    random_seed = random_seed
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset = InferenceDataset(root=data_root,
                               transform=transform)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    print('Finished data process!')
    return data_prefetcher(dataloader)


def tensor2im(var):
    var = var.cpu().detach().numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')
