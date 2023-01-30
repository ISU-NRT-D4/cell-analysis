import os

MASKS_DIRECTORY = os.path.join("data/IDCIA/", "ground_truth")
IMAGES_DIRECTORY = os.path.join("data/IDCIA/", "images")
BATCH_SIZE = 16

import albumentations as A
import albumentations.augmentations.functional as F

train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),

        A.Resize(height=256, width=256, always_apply=True),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Normalize()
    ]
)
val_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256,
                      always_apply=True, border_mode=0),
        A.Resize(height=256, width=256, always_apply=True),
        A.Normalize()
    ]
)
test_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256,
                      always_apply=True, border_mode=0),
        A.Resize(height=256, width=256, always_apply=True),
        A.Normalize()
    ]
)

lr = 1e-3
experiments= 5