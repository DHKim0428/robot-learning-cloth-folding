import torch
from datasets import load_dataset
import numpy as np

# load dataset
ds = load_dataset(
    "robot-learning-team43/so101_teleop_private",
    split="train"
)

# load model
checkpoint = torch.load("policy.pt", weights_only=False)

state_mean = torch.tensor(checkpoint["state_mean"], dtype=torch.float32)
state_std = torch.tensor(checkpoint["state_std"], dtype=torch.float32)

action_mean = torch.tensor(checkpoint["action_mean"], dtype=torch.float32)
action_std = torch.tensor(checkpoint["action_std"], dtype=torch.float32)

# rebuild model
state_dim = len(ds[0]["observation.state"])
action_dim = len(ds[0]["action"])

model = torch.nn.Sequential(
    torch.nn.Linear(state_dim, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, action_dim)
)

model.load_state_dict(checkpoint["model"])
model.eval()

# test on random sample
sample = ds[100]

state = torch.tensor(sample["observation.state"], dtype=torch.float32)
state = (state - state_mean) / state_std

pred = model(state.unsqueeze(0)).detach().numpy()[0]

# denormalize
pred = pred * action_std.numpy() + action_mean.numpy()

print("\nGround truth:", sample["action"])
print("Prediction   :", pred)