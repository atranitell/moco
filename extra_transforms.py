
import torch
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RGBtoYUV709F():

    def __init__(self):
        super().__init__()
    
    def __call__(self, img):
        """require input in [0, 1.0]
        """
        assert isinstance(img, torch.Tensor)

        img = img * 255.0
        
        R, G, B = torch.split(img, 1, dim=-3)
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B  # [0, 255]
        U = -0.1146 * R - 0.3854 * G + 0.5000 * B + 128  # [0, 255]
        V = 0.5000 * R - 0.4542 * G - 0.0468 * B + 128  # [0, 255]
        img = torch.cat([Y, U, V], dim=-3) / 255.0

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class YUV709FtoRGB():
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, img):
        """require input in [0, 1.0]
        """
        assert isinstance(img, torch.Tensor)
        img = img * 255.0
        Y, U, V = torch.split(img, 1, dim=-3)
        Y = Y
        U = U - 128
        V = V - 128
        R = 1.000 * Y + 1.570 * V
        G = 1.000 * Y - 0.187 * U - 0.467 * V
        B = 1.000 * Y + 1.856 * U
        return torch.cat([R, G, B], dim=-3) / 255.0

    def __repr__(self):
        return f"{self.__class__.__name__}()"