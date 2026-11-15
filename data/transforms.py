"""Training/validation transforms, including adverse-condition augmentation.

There is deliberately NO Normalize here. torchvision's detection models wrap
their backbone in a GeneralizedRCNNTransform that already applies ImageNet
mean/std (and resizing) internally. Normalizing here as well pushed every image
through those statistics twice, so the pretrained backbone saw inputs far outside
the distribution it was trained on. If you add a Normalize back, you must also
strip the model's internal one - not both.

Adverse conditions
------------------
"Can we still detect it in bad conditions" was a goal of this project from the
start, but nothing in the pipeline addressed it: the only augmentation was a
horizontal flip, and every scraped training image is a clear-weather publicity
photo. A model trained on those has never seen fog, rain, dusk or motion blur,
and there was no way to even measure how far it degrades.

Two things are provided here.

`get_train_transforms` applies degradations at random during training, re-rolled
every epoch. This is why the corpus's offline `_aug` chains were stripped rather
than reused: a file named `bird_0_aug0_aug0.jpg` is one fixed degradation baked
onto disk, counted as a separate image, and - having been generated before any
split existed - liable to put transformed copies of a training image into
validation. Doing it here instead means unlimited variety, no storage cost, and
no leakage.

`ADVERSE_SUITE` exposes each condition as a *named, deterministic* transform so
evaluation can report mAP per condition. That turns the original question from a
hope into a measurement: clear vs fog vs rain vs night, same weights, same
images.

Geometry is left alone by every degradation. These are all photometric or blur
operations, so boxes stay valid without being recomputed - the one exception is
the horizontal flip, which does transform its boxes.
"""

import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

_NUMPY_RNG = np.random.default_rng()


def set_transform_seed(seed):
    """Seed the float32 noise generator used by stochastic augmentations."""
    global _NUMPY_RNG
    _NUMPY_RNG = np.random.default_rng(seed)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            w = image.shape[2]
            boxes = target["boxes"].clone()
            if len(boxes) > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


class ResizeLongestSide:
    """Downscale oversized inputs before float-heavy augmentation.

    Faster R-CNN will cap the long side internally anyway. Doing that cap first
    prevents a 30-megapixel source photograph from creating several equally
    large float32 NumPy workspaces in fog/noise/gamma transforms. Boxes are
    scaled with the image, so geometry remains exact.
    """

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image, target):
        height, width = image.shape[-2:]
        longest = max(height, width)
        if longest <= self.max_size:
            return image, target
        scale = self.max_size / float(longest)
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))
        image = F.resize(image, [new_height, new_width], antialias=True)
        boxes = target["boxes"].clone()
        if len(boxes):
            boxes[:, [0, 2]] *= new_width / float(width)
            boxes[:, [1, 3]] *= new_height / float(height)
            target["boxes"] = boxes
        return image, target


# --------------------------------------------------------------------------
# Adverse conditions. Each takes and returns a float CHW tensor in [0, 1] and
# leaves `target` untouched.
# --------------------------------------------------------------------------

def _to_numpy(image):
    """CHW float tensor in [0,1] -> HWC float32 array in [0,1]."""
    return image.permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _to_tensor(arr):
    # Most augmentation outputs are already writable float32 arrays.  Clipping
    # and casting through temporaries doubles peak host memory and can fail on
    # Windows when the GPU driver's allocations have consumed the commit limit.
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags.writeable:
        arr = arr.copy()
    np.clip(arr, 0.0, 1.0, out=arr)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr).permute(2, 0, 1)


class Fog:
    """Atmospheric scattering: contrast collapses toward a bright airlight.

    Follows the standard haze model I = I0 * t + A * (1 - t), with transmission
    t falling off with a vertical gradient, since haze thickens toward the
    horizon in real aerial imagery.
    """

    def __init__(self, intensity=(0.3, 0.7), airlight=(0.75, 0.95)):
        self.intensity = intensity
        self.airlight = airlight

    def __call__(self, image, target):
        arr = _to_numpy(image)
        h, w = arr.shape[:2]
        beta = random.uniform(*self.intensity)
        A = random.uniform(*self.airlight)
        # Depth proxy: larger toward the top of the frame (further away).
        depth = np.linspace(1.0, 0.35, h, dtype=np.float32)[:, None]
        t = np.exp(-beta * 3.0 * depth)[:, :, None]
        return _to_tensor(arr * t + A * (1.0 - t)), target


class Rain:
    """Streaks plus the veiling glare and slight blur that rain adds."""

    def __init__(self, drops=(400, 1400), length=(8, 22), angle=(-20, 20),
                 brightness=(0.6, 0.9)):
        self.drops, self.length = drops, length
        self.angle, self.brightness = angle, brightness

    def __call__(self, image, target):
        arr = _to_numpy(image)
        h, w = arr.shape[:2]
        layer = np.zeros((h, w), dtype=np.float32)
        n = random.randint(*self.drops)
        L = random.randint(*self.length)
        theta = np.deg2rad(random.uniform(*self.angle))
        dx, dy = int(round(L * np.sin(theta))), int(round(L * np.cos(theta)))
        xs = np.random.randint(0, w, n)
        ys = np.random.randint(0, h, n)
        for x, y in zip(xs, ys):
            cv2.line(layer, (x, y), (x + dx, y + dy), 1.0, 1)
        layer = cv2.GaussianBlur(layer, (3, 3), 0)
        arr = arr * random.uniform(*self.brightness)
        arr = arr + layer[:, :, None] * 0.8
        # Rain on a lens also smears the scene slightly.
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
        return _to_tensor(arr), target


class LowLight:
    """Dusk/night: gamma darkening plus the sensor noise that comes with gain."""

    def __init__(self, gamma=(1.8, 3.2), gain=(0.35, 0.7), noise=(0.01, 0.045)):
        self.gamma, self.gain, self.noise = gamma, gain, noise

    def __call__(self, image, target):
        arr = _to_numpy(image)
        np.power(arr, random.uniform(*self.gamma), out=arr)
        arr *= np.float32(random.uniform(*self.gain))
        sigma = random.uniform(*self.noise)
        # ``np.random.normal`` allocates float64 first and only then casts it,
        # briefly doubling augmentation memory.  That is enough to exhaust the
        # Windows commit limit during a long detector run.  Generator supports
        # native float32 samples, so no hidden float64 buffer is created.
        noise = _NUMPY_RNG.standard_normal(arr.shape, dtype=np.float32)
        noise *= np.float32(sigma)
        arr += noise
        return _to_tensor(arr), target


class MotionBlur:
    """Directional blur from a moving target or an unstable camera."""

    def __init__(self, kernel=(5, 17), angle=(0, 180)):
        self.kernel, self.angle = kernel, angle

    def __call__(self, image, target):
        arr = _to_numpy(image)
        k = random.randrange(self.kernel[0] | 1, self.kernel[1] | 1, 2)
        kern = np.zeros((k, k), dtype=np.float32)
        kern[k // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5),
                                    random.uniform(*self.angle), 1.0)
        kern = cv2.warpAffine(kern, M, (k, k))
        s = kern.sum()
        if s > 0:
            kern /= s
        return _to_tensor(cv2.filter2D(arr, -1, kern)), target


class Overexposure:
    """Sun glare / white sky, where the airframe silhouette nearly washes out."""

    def __init__(self, gain=(1.3, 2.1), gamma=(0.45, 0.8)):
        self.gain, self.gamma = gain, gamma

    def __call__(self, image, target):
        arr = _to_numpy(image)
        arr = np.power(arr, random.uniform(*self.gamma))
        return _to_tensor(arr * random.uniform(*self.gain)), target


class SensorNoise:
    """Long-range EO/IR sensor grain, without the darkening of LowLight."""

    def __init__(self, sigma=(0.02, 0.07)):
        self.sigma = sigma

    def __call__(self, image, target):
        arr = _to_numpy(image)
        noise = _NUMPY_RNG.standard_normal(arr.shape, dtype=np.float32)
        arr = arr + noise * np.float32(random.uniform(*self.sigma))
        return _to_tensor(arr), target


class JPEGArtifacts:
    """Recompression blocking, as in relayed or downlinked video stills."""

    def __init__(self, quality=(18, 45)):
        self.quality = quality

    def __call__(self, image, target):
        arr = (_to_numpy(image) * 255).astype(np.uint8)
        q = random.randint(*self.quality)
        ok, buf = cv2.imencode(".jpg", arr[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return image, target
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
        return _to_tensor(dec.astype(np.float32) / 255.0), target


class ColorJitter:
    """Mild brightness/contrast/saturation drift - time of day and white balance."""

    def __init__(self, brightness=0.25, contrast=0.25, saturation=0.25):
        self.brightness, self.contrast = brightness, contrast
        self.saturation = saturation

    def __call__(self, image, target):
        if self.brightness:
            image = F.adjust_brightness(
                image, 1.0 + random.uniform(-self.brightness, self.brightness))
        if self.contrast:
            image = F.adjust_contrast(
                image, 1.0 + random.uniform(-self.contrast, self.contrast))
        if self.saturation:
            image = F.adjust_saturation(
                image, 1.0 + random.uniform(-self.saturation, self.saturation))
        return image.clamp(0.0, 1.0), target


class RandomApply:
    """Apply one transform with probability p."""

    def __init__(self, transform, p=0.5):
        self.transform, self.p = transform, p

    def __call__(self, image, target):
        if random.random() < self.p:
            return self.transform(image, target)
        return image, target


class RandomChoice:
    """Apply exactly one of `transforms` with probability p, else pass through.

    Conditions are chosen rather than stacked: fog on top of night on top of
    motion blur produces an image with no recoverable signal, and training on
    those teaches the model nothing except to predict nothing.
    """

    def __init__(self, transforms, p=0.5, weights=None):
        self.transforms, self.p, self.weights = transforms, p, weights

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        t = random.choices(self.transforms, weights=self.weights, k=1)[0]
        return t(image, target)


# Named, single-condition suites for evaluation. Deterministic in kind (each
# entry always applies its own condition), random only in severity.
ADVERSE_SUITE = {
    "clear": Compose([]),
    "fog": Compose([ResizeLongestSide(), Fog()]),
    "rain": Compose([ResizeLongestSide(), Rain()]),
    "low_light": Compose([ResizeLongestSide(), LowLight()]),
    "motion_blur": Compose([ResizeLongestSide(), MotionBlur()]),
    "overexposure": Compose([ResizeLongestSide(), Overexposure()]),
    "sensor_noise": Compose([ResizeLongestSide(), SensorNoise()]),
    "jpeg_artifacts": Compose([ResizeLongestSide(), JPEGArtifacts()]),
}


def get_train_transforms(adverse_prob=0.5, jitter=True):
    """Training transforms.

    `adverse_prob` is the chance that any one degradation is applied. 0.5 keeps
    half the batch clean, which matters: the detector still has to work in good
    conditions, and training only on degraded images costs clear-weather
    accuracy.
    """
    transforms = [ResizeLongestSide(), RandomHorizontalFlip(0.5)]
    if jitter:
        transforms.append(RandomApply(ColorJitter(), p=0.5))
    transforms.append(RandomChoice(
        [Fog(), Rain(), LowLight(), MotionBlur(), Overexposure(),
         SensorNoise(), JPEGArtifacts()],
        p=adverse_prob,
        # Fog, low light and motion blur are the conditions a fielded detector
        # actually faces most, so they are sampled more often.
        weights=[1.5, 1.2, 1.5, 1.5, 0.8, 1.0, 0.8],
    ))
    return Compose(transforms)


def get_val_transforms():
    # Match the deterministic geometry used before training augmentation.  The
    # detector would resize internally, but only after the dataset has created
    # and transferred a full-resolution float tensor.  Pre-capping the long
    # side keeps multi-megapixel evaluation images from exhausting GPU/Windows
    # commit memory while preserving boxes exactly.
    return Compose([ResizeLongestSide()])


def get_adverse_transforms(condition):
    """Return the named evaluation transform for one condition."""
    try:
        return ADVERSE_SUITE[condition]
    except KeyError:
        raise KeyError(f"unknown condition {condition!r}; "
                       f"known: {sorted(ADVERSE_SUITE)}") from None
