import albumentations as A


def build_train_augmentations() -> list:
    return [
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_range=(0, 1),
            num_flare_circles_range=(6, 10),
            p=0.25,
        ),
        A.RandomFog(
            fog_coef_range=(0.2, 0.5),
            alpha_coef=0.08,
            p=0.25,
        ),
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=20,
            drop_width=1,
            blur_value=3,
            brightness_coefficient=0.8,
            rain_type="default",
            p=0.25,
        ),
    ]
