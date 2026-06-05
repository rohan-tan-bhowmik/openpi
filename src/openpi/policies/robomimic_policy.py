import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robomimic_example() -> dict:
    """Creates a random input example for the RoboMimic policy."""
    return {
        "observation/state": np.random.rand(9),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "insert the peg into the square hole",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboMimicInputs(transforms.DataTransformFn):
    """Converts RoboMimic observations into the model's expected input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        if "observation/wrist_image" in data:
            wrist_image = _parse_image(data["observation/wrist_image"])
            wrist_mask = np.True_
        else:
            wrist_image = np.zeros_like(base_image)
            wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # RoboMimic only provides one wrist camera, so mirror training behavior on the second slot.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboMimicOutputs(transforms.DataTransformFn):
    """Converts model outputs back to RoboMimic's 7D action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
