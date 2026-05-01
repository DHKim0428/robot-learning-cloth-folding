from datasets import load_dataset
import torch
import numpy as np
import random

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

# evaluate
errors = []

for _ in range(1000):
    sample = ds[random.randint(0, len(ds)-1)]

    state = torch.tensor(sample["observation.state"], dtype=torch.float32)
    action = torch.tensor(sample["action"], dtype=torch.float32)

    state = (state - state_mean) / state_std

    pred = model(state.unsqueeze(0)).detach()[0]
    pred = pred * action_std + action_mean

    error = torch.norm(pred - action)
    errors.append(error.item())

print("Mean error:", np.mean(errors))
print("Std error :", np.std(errors))