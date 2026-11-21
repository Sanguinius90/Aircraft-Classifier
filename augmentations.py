import albumentations as A


def build_train_augmentations() -> list:
    """Build a conservative RandAugment-style appearance policy."""
    operation_pool = [
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=1.0,
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=1.0,
        ),
        A.CLAHE(
            clip_limit=(1.0, 2.0),
            tile_grid_size=(8, 8),
            p=1.0,
        ),
        A.Sharpen(
            alpha=(0.1, 0.2),
            lightness=(0.8, 1.0),
            p=1.0,
        ),
        A.RandomGamma(
            gamma_limit=(85, 115),
            p=1.0,
        ),
        A.ISONoise(
            color_shift=(0.01, 0.03),
            intensity=(0.05, 0.15),
            p=1.0,
        ),
    ]

    return [
        A.RandomOrder(
            transforms=operation_pool,
            n=2,
            replace=False,
            p=1.0,
        )
    ]


def build_light_augmentations() -> list:
    """Fallback used by the previous light weather-augmentation training."""
    return [
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_range=(0, 1),
            num_flare_circles_range=(6, 10),
            p=0.1,
        ),
        A.RandomFog(
            fog_coef_range=(0.2, 0.5),
            alpha_coef=0.08,
            p=0.1,
        ),
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=20,
            drop_width=1,
            blur_value=3,
            brightness_coefficient=0.8,
            rain_type="default",
            p=0.1,
        ),
    ]
