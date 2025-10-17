import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import cfg

def scale_to_bounds(x, y):
    sx = torch.sigmoid(x)
    sy = torch.sigmoid(y)
    out_x = cfg.x_min + sx * (cfg.x_max - cfg.x_min)
    out_y = cfg.y_min + sy * (cfg.y_max - cfg.y_min)
    return out_x, out_y

# Actor
class Actor(nn.Module):
    """
    (x, y) -> sigmoid -> [x_min, x_max], [y_min, y_max]
    """
    def __init__(self,
                 state_dim=cfg.obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, cfg.act_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        out_x, out_y = out[:, 0], out[:, 1]
        scaled_x, scaled_y = scale_to_bounds(out_x, out_y)
        return torch.stack([scaled_x, scaled_y], dim=1)

# Critic
class Critic(nn.Module):
    """
    s: (B, OBS_DIM), a: (B, 2)
    """
    def __init__(self,
                 state_dim=cfg.obs_dim,
                 action_dim=cfg.act_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)    # Q(s,a)

    def forward(self, s, a):
        """
        :param s: (B, OBS_DIM)
        :param a: (B, 2)
        :return: Q(s, a)
        """
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

# TD3 Agent
class TD3Agent:
    def __init__(self, device="cuda:0"):
        self.device = device

        self.actor = Actor().to(device)
        self.critic1 = Critic().to(device)
        self.critic2 = Critic().to(device)

        self.actor_target = Actor().to(device)
        self.critic1_target = Critic().to(device)
        self.critic2_target = Critic().to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=cfg.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=cfg.lr_critic)

        # Replay buffer
        self.replay_buffer = deque(maxlen=cfg.replay_size)

        # For delayed policy updates
        self.policy_update_count = 0

    def select_action(self, state):
        s_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            a_t = self.actor(s_t.unsqueeze(0))
        return a_t.squeeze(0).cpu().numpy()

    def store_transition(self, s, a, r, s_next):
        self.replay_buffer.append((s, a, r, s_next))

    def _soft_update(self, net, net_target):
        for param, param_tgt in zip(net.parameters(), net_target.parameters()):
            param_tgt.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * param_tgt.data)

    def update(self, gradient_steps=1):
        if len(self.replay_buffer) < cfg.replay_size:
            print(f"[UPDATE] Replay buffer size is {len(self.replay_buffer)}")
            return None, None, None

        batch = random.sample(self.replay_buffer, cfg.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(np.array(rewards))
        next_states = np.array(next_states)

        states_t = torch.FloatTensor(states).to(self.device)            # (B, OBS_DIM)
        actions_t = torch.FloatTensor(actions).to(self.device)          # (B, 2)
        rewards_t = torch.FloatTensor(rewards).to(self.device)          # (B, 1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)  # (B, OBS_DIM)

        with torch.no_grad():
            next_action = self.actor_target(next_states_t)              # (B, 2)

            # target policy smoothing
            noise = (torch.rand_like(next_action) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_action = next_action + noise

            next_action_x = next_action[:, 0].clamp(cfg.x_min, cfg.x_max)
            next_action_y = next_action[:, 1].clamp(cfg.y_min, cfg.y_max)
            next_action = torch.stack([next_action_x, next_action_y], dim=1)

            q1_target = self.critic1_target(next_states_t, next_action)
            q2_target = self.critic2_target(next_states_t, next_action)
            q_min = torch.min(q1_target, q2_target)

            # TD target
            # y = r + gamma * q_min
            target_q = rewards_t + cfg.gamma * q_min

        q1 = self.critic1(states_t, actions_t)
        q2 = self.critic2(states_t, actions_t)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        critic_loss = critic1_loss + critic2_loss
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        self.policy_update_count += 1
        if self.policy_update_count % cfg.policy_delay == 0:
            print("[UPDATE] Actor loss update")
            actor_actions = self.actor(states_t)
            actor_loss = -self.critic1(states_t, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

            return critic1_loss, critic2_loss, actor_loss
        return critic1_loss, critic2_loss, torch.tensor(torch.inf)