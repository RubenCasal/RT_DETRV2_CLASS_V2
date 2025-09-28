# Data Augmentation — Albumentations

This document explains the **data augmentation pipeline** used in the project. Data augmentations help to improve model generalization by simulating diverse real-world conditions such as lighting, weather, and camera distortions.

The augmentations are defined in a constant called `AUGMENTATIONS`, which contains a list of transformations provided by the **Albumentations** library.

---

## Augmentations Overview

| Transformation          | Purpose                                                            | Key Parameters                             |
| ----------------------- | ------------------------------------------------------------------ | ------------------------------------------ |
| **Affine**              | Scale, rotate, shear, translate. Simulates camera distortions.     | scale, rotate, shear                       |
| **Gaussian Noise**      | Adds random noise. Improves robustness to low-light/sensor issues. | std_range                                  |
| **Brightness/Contrast** | Adjusts brightness & contrast. Mimics lighting variation.          | brightness_limit, contrast_limit           |
| **Motion Blur**         | Slight blur for motion simulation.                                 | blur_limit                                 |
| **Random Shadow**       | Semi-transparent shadows to simulate occlusion.                    | shadow_roi, shadow_intensity_range         |
| **Random Sun Flare**    | Simulates sunlight glare in top region.                            | src_radius, num_flare_circles_range        |
| **Random Fog**          | Reduces visibility to simulate foggy scenes.                       | fog_coef_lower, fog_coef_upper             |
| **Random Rain**         | Adds rain streaks + brightness reduction.                          | slant, drop_length, brightness_coefficient |

---

## Implementation

```python
AUGMENTATIONS = [
    A.Affine(
        scale=(0.90, 1.10),
        translate_percent=(0.0, 0.05),
        rotate=(-10, 10),
        shear=(-4, 4),
        p=0.5
    ),

    A.AdditiveNoise(
        noise_type="gaussian",
        spatial_mode="shared",
        noise_params={"mean_range": [0, 0], "std_range": [0.01, 0.03]},
        approximation=1
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.1,
        p=0.3
    ),

    A.MotionBlur(
        blur_limit=2,
        p=0.01
    ),

    A.RandomShadow(
        shadow_roi=[0, 0.5, 1, 1],
        num_shadows_limit=[2, 3],
        shadow_dimension=4,
        shadow_intensity_range=[0.15, 0.5],
        p=0.3
    ),

    A.RandomSunFlare(
        flare_roi=[0, 0, 1, 0.5],
        src_radius=250,
        src_color=[255, 255, 255],
        angle_range=[0, 1],
        num_flare_circles_range=[3, 6],
        method="overlay",
        p=0.1
    ),

    A.RandomFog(
        fog_coef_lower=0.05,
        fog_coef_upper=0.2,
        alpha_coef=0.05,
        p=0.01
    ),

    A.RandomRain(
        slant_lower=-5,
        slant_upper=5,
        drop_length=10,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=3,
        brightness_coefficient=0.95,
        p=0.05
    ),
]
```

---
