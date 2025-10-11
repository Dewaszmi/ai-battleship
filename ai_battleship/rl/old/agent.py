import numpy as np
import torch

from ai_battleship.rl.old.model import ActorCritic


class RolloutBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.device = device
        self.capacity = capacity
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.logps = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.masks = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, action, logp, reward, value, mask):
        self.obs[self.ptr].copy_(obs)
        self.actions[self.ptr] = action
        self.logps[self.ptr] = logp
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.masks[self.ptr] = mask
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        ptr = self.ptr
        adv = torch.zeros(ptr, device=self.device)
        last_gae = 0.0
        prev_value = last_value
        for step in reversed(range(ptr)):
            mask = self.masks[step]
            delta = self.rewards[step] + gamma * prev_value * mask - self.values[step]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[step] = last_gae
            prev_value = self.values[step]
        returns = adv + self.values[:ptr]
        self.advantages = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        self.returns = returns
        self.size = ptr

    def get_batches(self, batch_size):
        idxs = np.arange(self.size)
        np.random.shuffle(idxs)
        for start in range(0, self.size, batch_size):
            b = idxs[start : start + batch_size]
            yield (
                self.obs[b],
                self.actions[b],
                self.logps[b],
                self.returns[b],
                self.advantages[b],
            )

    def clear(self):
        self.ptr = 0


class Agent:
    def __init__(
        self,
        model: ActorCritic,
        lr: float = 1e-4,
        clip_eps: float = 0.1,
        value_coef: float = 0.5,
        ent_coef: float = 0.2,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.ent_coef = ent_coef

    def update(self, buffer: RolloutBuffer, epochs: int = 4, batch_size: int = 32):
        for _ in range(epochs):
            for obs_b, actions_b, old_logp_b, returns_b, adv_b in buffer.get_batches(batch_size):
                obs_b = obs_b.to(self.device)
                actions_b = actions_b.to(self.device)
                old_logp_b = old_logp_b.to(self.device)
                returns_b = returns_b.to(self.device)
                adv_b = adv_b.to(self.device)

                logp, value, ent = self.model.get_logp_value_entropy(obs_b, actions_b)
                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns_b - value).pow(2).mean()
                entropy_loss = -ent.mean()

                loss = policy_loss + self.value_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
