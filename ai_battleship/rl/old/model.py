import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, in_channels: int, grid_size: int, hidden_dim: int = 256):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * grid_size * grid_size
        self.fc = nn.Sequential(nn.Linear(conv_out, hidden_dim), nn.ReLU())
        self.policy = nn.Linear(hidden_dim, grid_size * grid_size)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value, dist.entropy()

    def get_logp_value_entropy(self, x, actions):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ent = dist.entropy()
        return logp, value, ent
