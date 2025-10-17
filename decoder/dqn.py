import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

from config import cfg

class DQNNet(nn.Module):
    def __init__(self,
                 in_dim=cfg.obs_dim,
                 out_dim=cfg.dqn_act_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, device="cuda"):
        self.device = device
        self.q_net = DQNNet().to(self.device)
        self.q_target = DQNNet().to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.dqn_lr)

        self.replay_buffer = deque(maxlen=cfg.replay_size)

        # Epsilon 관련 설정
        self.eps_start = 1.0
        self.eps_min = 0.05
        self.eps = self.eps_start

        # Epoch 기반 decay 설정
        self.current_epoch = 0
        self.total_epochs = cfg.epochs if hasattr(cfg, 'epochs') else 100
        self.eps_decay_strategy = cfg.eps_decay_strategy if hasattr(cfg, 'eps_decay_strategy') else 'linear'

        # Step 기반 업데이트 설정
        self.update_interval = 10
        self.step_count = 0

        # self.eps = 1.0
        # self.eps_min = 0.05
        # self.eps_decay = 1e-3
        # self.update_interval = 10
        # self.step_count = 0

    def set_epoch(self, epoch, total_epochs=None):
        self.current_epoch = epoch
        if total_epochs:
            self.total_epochs = total_epochs

            # Epsilon decay 전략에 따라 업데이트
            if self.eps_decay_strategy == 'linear':
                # Linear decay: epoch에 비례하여 선형 감소
                decay_ratio = epoch / self.total_epochs
                self.eps = self.eps_start - (self.eps_start - self.eps_min) * decay_ratio

            elif self.eps_decay_strategy == 'exponential':
                # Exponential decay: 지수적 감소
                decay_rate = 0.95  # 각 epoch마다 95%로 감소
                self.eps = max(self.eps_min, self.eps_start * (decay_rate ** epoch))

            elif self.eps_decay_strategy == 'cosine':
                # Cosine annealing: 코사인 함수를 이용한 부드러운 감소
                self.eps = self.eps_min + (self.eps_start - self.eps_min) * \
                           (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2

            elif self.eps_decay_strategy == 'step':
                # Step decay: 특정 epoch마다 급격히 감소
                step_size = self.total_epochs // 4  # 4단계로 나눔
                decay_rate = 0.5
                num_steps = epoch // step_size
                self.eps = max(self.eps_min, self.eps_start * (decay_rate ** num_steps))

            elif self.eps_decay_strategy == 'warmup_cosine':
                # Warmup + Cosine: 초기에는 높은 exploration, 이후 cosine decay
                warmup_epochs = self.total_epochs // 10  # 처음 10%는 warmup
                if epoch < warmup_epochs:
                    self.eps = self.eps_start
                else:
                    adjusted_epoch = epoch - warmup_epochs
                    adjusted_total = self.total_epochs - warmup_epochs
                    self.eps = self.eps_min + (self.eps_start - self.eps_min) * \
                               (1 + np.cos(np.pi * adjusted_epoch / adjusted_total)) / 2

            # 최소값 보장
            self.eps = max(self.eps, self.eps_min)

    def get_epsilon_info(self):
        return {
            'current_epsilon': self.eps,
            'epoch': self.current_epoch,
            'strategy': self.eps_decay_strategy
        }

    def select_action(self, obs, invalid_actions=None):
        if random.random() < self.eps:
            return random.randint(0, cfg.dqn_act_dim - 1)
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_vals = self.q_net(obs_t)  # shape: (1 ,13)

                q_vals = q_vals.squeeze(0)  # shape (13,)

                # Q values of invalid actions are -inf
                masked_q_vals = q_vals.clone()
                for invalid_action in invalid_actions:
                    masked_q_vals[invalid_action] = float('-inf')
                return int(masked_q_vals.argmax().item())

    def store_transition(self, s, a, r, s_next):
        self.replay_buffer.append((s, a, r, s_next))

    def update(self):
        if len(self.replay_buffer) < cfg.batch_size:
            print(f"[UPDATE] Skipped - insufficient data {len(self.replay_buffer)}/{cfg.batch_size} in replay buffer")
            return

        batch = random.sample(self.replay_buffer, cfg.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states_t = torch.FloatTensor(states).to(self.device)            # (B, OBS_DIM)
        actions_t = torch.LongTensor(actions).to(self.device)           # (B,)
        rewards_t = torch.FloatTensor(rewards).to(self.device)          # (B,)
        next_states_t = torch.FloatTensor(next_states).to(self.device)

        # Q(s, a)
        q_all = self.q_net(states_t)                                    # (B, ACTION_DIM)
        q_sa = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)       # (B,)

        # target = r + gamma * maxQ(s',a')
        with torch.no_grad():
            q_next_all = self.q_target(next_states_t)                   # (B, ACTION_DIM)
            q_next_max, _ = q_next_all.max(dim=1)                       # (B,)
            target = rewards_t + cfg.gamma * q_next_max

        loss = F.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_interval == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return loss