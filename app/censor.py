import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from app import config

DIRECT_LIMIT = 61
TARGET_SIGMA = 3.0

def expand_box(box, padding, width, height):
    x, y, w, h = box
    left = int(x) - int(padding)
    top = int(y) - int(padding)
    right = int(x) + int(w) + int(padding)
    bottom = int(y) + int(h) + int(padding)
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom

class BlurEngine:
    def __init__(self, device="cpu", strength=None):
        self.device = torch.device(device if device in ("cuda", "mps") else "cpu")
        self.use_torch = self.device.type in ("cuda", "mps")
        self.strength = 0
        self.set_strength(strength or config.BLUR_STRENGTH)

    def set_strength(self, strength):
        size = int(strength)
        if size < 3:
            size = 3
        if size > config.BLUR_MAX:
            size = config.BLUR_MAX
        if size % 2 == 0:
            size += 1
        if size == self.strength:
            return
        self.strength = size
        self.sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
        if size > DIRECT_LIMIT:
            self.factor = max(1, int(round(self.sigma / TARGET_SIGMA)))
        else:
            self.factor = 1
        self.working_sigma = self.sigma / float(self.factor)
        self.kernel_size = int(2 * round(3.0 * self.working_sigma) + 1)
        coords = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) / 2.0
        kernel = torch.exp(-(coords * coords) / (2.0 * self.working_sigma * self.working_sigma))
        kernel = kernel / kernel.sum()
        if self.use_torch:
            self.kernel_h = kernel.view(1, 1, 1, self.kernel_size).repeat(3, 1, 1, 1).to(self.device)
            self.kernel_v = kernel.view(1, 1, self.kernel_size, 1).repeat(3, 1, 1, 1).to(self.device)

    def working_size(self, height, width):
        if self.factor <= 1:
            return height, width
        return max(1, -(-height // self.factor)), max(1, -(-width // self.factor))

    def _blur_torch(self, crop):
        height, width = crop.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(crop)).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        small_height, small_width = self.working_size(height, width)
        if (small_height, small_width) != (height, width):
            tensor = functional.avg_pool2d(tensor, kernel_size=self.factor, stride=self.factor, ceil_mode=True)
        pad = self.kernel_size // 2
        pad_h = min(pad, small_width - 1)
        pad_v = min(pad, small_height - 1)
        if pad_h > 0:
            tensor = functional.pad(tensor, (pad_h, pad_h, 0, 0), mode="replicate")
            kernel = self.kernel_h[:, :, :, pad - pad_h : pad + pad_h + 1]
            tensor = functional.conv2d(tensor, kernel / kernel.sum(dim=3, keepdim=True), groups=3)
        if pad_v > 0:
            tensor = functional.pad(tensor, (0, 0, pad_v, pad_v), mode="replicate")
            kernel = self.kernel_v[:, :, pad - pad_v : pad + pad_v + 1, :]
            tensor = functional.conv2d(tensor, kernel / kernel.sum(dim=2, keepdim=True), groups=3)
        if (small_height, small_width) != (height, width):
            tensor = functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        output = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
        return output.cpu().numpy()

    def _blur_cpu(self, crop):
        height, width = crop.shape[:2]
        small_height, small_width = self.working_size(height, width)
        working = crop
        if (small_height, small_width) != (height, width):
            working = cv2.resize(crop, (small_width, small_height), interpolation=cv2.INTER_AREA)
        limit_h = min(self.kernel_size, small_width if small_width % 2 == 1 else small_width - 1)
        limit_v = min(self.kernel_size, small_height if small_height % 2 == 1 else small_height - 1)
        limit_h = max(3, limit_h if limit_h % 2 == 1 else limit_h - 1)
        limit_v = max(3, limit_v if limit_v % 2 == 1 else limit_v - 1)
        working = cv2.GaussianBlur(working, (limit_h, limit_v), self.working_sigma)
        if (small_height, small_width) != (height, width):
            working = cv2.resize(working, (width, height), interpolation=cv2.INTER_LINEAR)
        return working

    def blur_regions(self, frame, regions):
        if not regions:
            return frame
        for left, top, right, bottom in regions:
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                continue
            if self.use_torch:
                frame[top:bottom, left:right] = self._blur_torch(crop)
            else:
                frame[top:bottom, left:right] = self._blur_cpu(crop)
        return frame