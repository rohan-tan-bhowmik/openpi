import numpy as np

from openpi.models import model as _model
from openpi.policies import robomimic_policy


def test_robomimic_inputs_maps_observation_keys():
    transform = robomimic_policy.RoboMimicInputs(model_type=_model.ModelType.PI0)

    data = {
        "observation/state": np.random.rand(9).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "insert the peg into the square hole",
    }

    result = transform(data)

    assert result["state"].shape == (9,)
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert np.array_equal(result["image"]["left_wrist_0_rgb"], result["image"]["right_wrist_0_rgb"])
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
    }


def test_robomimic_inputs_pads_missing_wrist_image():
    transform = robomimic_policy.RoboMimicInputs(model_type=_model.ModelType.PI0)

    base_image = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    result = transform(
        {
            "observation/state": np.random.rand(9).astype(np.float32),
            "observation/image": base_image,
            "prompt": "insert the peg into the square hole",
        }
    )

    assert np.array_equal(result["image"]["left_wrist_0_rgb"], np.zeros_like(base_image))
    assert np.array_equal(result["image"]["right_wrist_0_rgb"], np.zeros_like(base_image))
    assert result["image_mask"]["left_wrist_0_rgb"] == np.False_
    assert result["image_mask"]["right_wrist_0_rgb"] == np.False_
