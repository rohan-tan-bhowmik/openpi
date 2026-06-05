import numpy as np
from openpi_client import websocket_client_policy


client = websocket_client_policy.WebsocketClientPolicy(
    host="127.0.0.1",
    port=8000,
)

# Fake observation matching the RoboMimic square-peg image policy.
observation = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros((9,), dtype=np.float32),
    "prompt": "insert the peg into the square hole",
}

print("Sending RoboMimic-format test request...")

try:
    result = client.infer(observation)
except Exception as exc:
    print("FAILED:", repr(exc))
    print("Debug shapes/dtypes:")
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    raise

print("SUCCESS")
print("keys:", result.keys())
print("actions dtype:", getattr(result["actions"], "dtype", None))
print("actions shape:", result["actions"].shape)
print("first action row:", result["actions"][0])
