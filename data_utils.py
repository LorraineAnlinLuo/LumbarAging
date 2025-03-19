import numpy as np
from box import Box
from torchvision.transforms import transforms
from lib.dataset import Lumbar
from torch.utils.data import DataLoader


def find_closest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def frame_drop(scan, frame_keep_style="random", frame_keep_fraction=1, frame_dim=1, impute="drop"):
    if frame_keep_fraction >= 1:
        return scan

    n = scan.shape[frame_dim]
    if frame_keep_style == "random":
        frames_to_keep = int(np.floor(n * frame_keep_fraction))
        indices = np.random.permutation(np.arange(n))[:frames_to_keep]
    elif frame_keep_style == "ordered":
        k = int(np.ceil(1 / frame_keep_fraction))
        indices = np.arange(0, n, k)
        # pick every k'th frame
    else:
        raise Exception("Wrong frame drop style")

    if impute == "zeros":
        if frame_dim == 1:
            t = np.zeros((1, n, 1, 1))
            t[:, indices] = 1
            return scan * t
        if frame_dim == 2:
            t = np.zeros((1, 1, n, 1))
            t[:, :, indices] = 1
            return scan * t
        if frame_dim == 3:
            t = np.zeros((1, 1, 1, n))
            t[:, :, :, indices] = 1
            return scan * t
    elif impute == "fill":
        # fill with nearest available frame
        if frame_dim == 1:
            for i in range(scan.shape[1]):
                if i in indices:
                    pass
                scan[:, i, :, :] = scan[:, find_closest(i, indices), :, :]
            return scan

        if frame_dim == 2:
            for i in range(scan.shape[1]):
                if i in indices:
                    pass
                scan[:, :, i, :] = scan[:, :, find_closest(i, indices), :]
            return scan

        if frame_dim == 3:
            for i in range(scan.shape[1]):
                if i in indices:
                    pass
                scan[:, :, :, i] = scan[:, :, :, find_closest(i, indices)]
            return scan
    elif impute == "noise":
        noise = np.random.uniform(high=scan.max(), low=scan.min(), size=scan.shape)

        if frame_dim == 1:
            t = np.zeros((1, n, 1, 1))
            t[:, indices] = 1
            return scan + noise * (1 - t)
        if frame_dim == 2:
            t = np.zeros((1, 1, n, 1))
            t[:, :, indices] = 1
            return scan + noise * (1 - t)
        if frame_dim == 3:
            t = np.zeros((1, 1, 1, n))
            t[:, :, :, indices] = 1
            return scan + noise * (1 - t)
    else: # drop
        if frame_dim == 1:
            return scan[:, indices, :, :]
        if frame_dim == 2:
            return scan[:, :, indices, :]
        if frame_dim == 3:
            return scan[:, :, :, indices]

    return scan


def gaussian_noise(scan, sigma=0.0):
    return scan + sigma * np.random.randn(*scan.shape)


def intensity_scaling(scan):
    scale = 2 ** (2 * np.random.rand() - 1)  # 2**(-1,1)
    return scan * scale

def get_loader(args):
    # Transformations to remove frames
    # frame_drop_transform = lambda x: frame_drop(x, frame_keep_style=args.frame_keep_style, frame_keep_fraction=args.frame_keep_fraction, frame_dim=args.frame_dim, impute=args.impute)
    # transform = transforms.Compose([frame_drop_transform])
    transform=None

    train_data = Lumbar(root=args.root_path, metadatafile=args.train_csv, transform=transform)
    val_data = Lumbar(root=args.root_path, metadatafile=args.val_csv, transform=transform)
    test_data = Lumbar(root=args.root_path, metadatafile=args.test_csv, transform=transform)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_loader,val_loader,test_loader