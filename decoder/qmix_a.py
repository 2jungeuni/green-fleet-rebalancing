import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from collections import deque

from config import cfg


class QAgentNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QMixer(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()

        self.attn_fc = nn.Linear(state_dim + 1, 1)

        self.fc1 = nn.Linear(state_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, agent_qs, state, mask=None):
        B, n_agents = agent_qs.shape

        state_expanded = state.unsqueeze(1).expand(-1, n_agents, -1)    # (B, n_agents, state_dim)

        q_expanded = agent_qs.unsqueeze(-1)                             # (B, n_agents, 1)

        attn_input = torch.cat([state_expanded, q_expanded], dim=-1)

        attn_logits = self.attn_fc(attn_input).squeeze(-1)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=1)                    # (B, n_agents)

        weighted_q = agent_qs * attn_weights
        q_agg = weighted_q.sum(dim=1, keepdim=True)                     # (B, 1)

        x = torch.cat([state, q_agg], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QMIXAgent:
    def __init__(self,
                 n_agents,
                 obs_dim,
                 state_dim,
                 act_dim,
                 replay_size=cfg.replay_size,
                 lr=cfg.dqn_lr,
                 gamma=cfg.gamma,
                 batch_size=cfg.batch_size,
                 device="cuda"):
        self.device = device

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.act_dim = act_dim

        # Epsilon 관련 설정
        self.eps_start = 1.0
        self.eps_min = 0.05
        self.eps = self.eps_start

        # Epoch 기반 decay 설정
        self.current_epoch = 0
        self.total_epochs = cfg.epochs if hasattr(cfg, 'epochs') else 100
        self.eps_decay_strategy = cfg.eps_decay_strategy if hasattr(cfg, 'eps_decay_strategy') else 'linear'

        self.agent_nets = nn.ModuleList([
            QAgentNet(self.obs_dim, self.act_dim) for _ in range(self.n_agents)
        ]).to(self.device)

        self.target_agent_nets = nn.ModuleList([
            QAgentNet(self.obs_dim, self.act_dim) for _ in range(self.n_agents)
        ]).to(self.device)

        for target_net, net in zip(self.target_agent_nets, self.agent_nets):
            target_net.load_state_dict(net.state_dict())

        # Attention Mixer & Target Mixer
        self.mixer = QMixer(self.state_dim).to(self.device)
        self.target_mixer = QMixer(self.state_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Optimizer
        self.parameters = list(self.agent_nets.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=lr)

        self.replay_buffer = deque(maxlen=replay_size)

        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 1e-3
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.update_interval = 10
        self.train_step = 0

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

    def select_actions(self, obs_list):
        actions = []
        active_agents = len(obs_list)
        active_agents = min(active_agents, self.n_agents)

        for i in range(active_agents):
            if random.random() < self.eps:
                a = random.randint(0, self.act_dim - 1)
            else:
                obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
                if i < len(self.agent_nets):
                    q_vals = self.agent_nets[i](obs_t)
                    a = int(q_vals.argmax(dim=1))
                else:
                    a = random.randint(0, self.act_dim - 1)
            actions.append(a)
        return actions

    def store_transition(self, state, obs_list, actions, reward, next_state, next_obs_list):
        self.replay_buffer.append((state, obs_list, actions, reward, next_state, next_obs_list))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)

        state_b, obs_b, actions_b, rewards_b, next_state_b, next_obs_b = zip(*batch)

        state_t = torch.FloatTensor(state_b).to(self.device)
        next_state_t = torch.FloatTensor(next_state_b).to(self.device)
        rewards_t = torch.FloatTensor(rewards_b).to(self.device)

        max_agents_in_batch = max(len(obs_list) for obs_list in obs_b)
        n_agents_in_batch = max_agents_in_batch

        # obs_t, actions_t, mask_t
        obs_t_list = []
        actions_t_list = []
        mask_t_list = []
        for b_idx in range(self.batch_size):
            active_agents = len(obs_b[b_idx])

            obs_pad = np.zeros((n_agents_in_batch, self.obs_dim), dtype=np.float32)
            act_pad = np.zeros((n_agents_in_batch,), dtype=np.int64)
            obs_pad[:active_agents] = np.array(obs_b[b_idx])
            act_pad[:active_agents] = np.array(actions_b[b_idx])

            obs_t_list.append(obs_pad)
            actions_t_list.append(act_pad)

            mask = np.zeros((n_agents_in_batch,), dtype=np.float32)
            mask[:active_agents] = 1.0
            mask_t_list.append(mask)

        obs_t = torch.FloatTensor(obs_t_list).to(self.device)           # (B, n_agents_in_batch, obs_dim)
        actions_t = torch.LongTensor(actions_t_list).to(self.device)    # (B, n_agents_in_batch)
        mask_t = torch.FloatTensor(mask_t_list).to(self.device)         # (B, n_agents_in_batch)

        # next_obs_t, next_mask_t
        next_obs_t_list = []
        next_mask_t_list = []
        for b_idx in range(self.batch_size):
            active_agents = len(next_obs_b[b_idx])
            obs_pad = np.zeros((n_agents_in_batch, self.obs_dim), dtype=np.float32)
            obs_pad[:active_agents] = np.array(next_obs_b[b_idx])
            next_obs_t_list.append(obs_pad)

            mask = np.zeros((n_agents_in_batch,), dtype=np.float32)
            mask[:active_agents] = 1.0
            next_mask_t_list.append(mask)

        next_obs_t = torch.FloatTensor(next_obs_t_list).to(self.device)
        next_mask_t = torch.FloatTensor(next_mask_t_list).to(self.device)

        all_q = []
        for i in range(n_agents_in_batch):
            q_i = self.agent_nets[i](obs_t[:, i, :])
            all_q.append(q_i.unsqueeze(1))
        all_q = torch.cat(all_q, dim=1)

        chosen_q = []
        for i in range(n_agents_in_batch):
            q_i = all_q[:, i, :]
            a_i = actions_t[:, i].unsqueeze(1)                      # (B, 1)
            q_i_chosen = q_i.gather(1, a_i).squeeze(1)          # (B,)
            chosen_q.append(q_i_chosen.unsqueeze(1))
        chosen_q = torch.cat(chosen_q, dim=1)                       # (B, n_agents_in_batch)

        q_total = self.mixer(chosen_q, state_t, mask_t).squeeze(-1)

        with torch.no_grad():
            target_q_list = []
            for i in range(n_agents_in_batch):
                q_i_next = self.target_agent_nets[i](next_obs_t[:, i, :])
                q_i_max = q_i_next.max(dim=1)[0].unsqueeze(1)
                target_q_list.append(q_i_max)

            target_q = torch.cat(target_q_list, dim=1)      # (B, n_agents_in_batch)

            q_total_next = self.target_mixer(target_q, next_state_t, next_mask_t).squeeze(-1)

            target_value = rewards_t + self.gamma * q_total_next

        loss = F.mse_loss(q_total, target_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.update_interval == 0:
            for target_net, net in zip(self.target_agent_nets, self.agent_nets):
                target_net.load_state_dict(net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Epsilon decay
        if self.eps > self.eps_min:
            self.eps -= self.eps_decay
            self.eps = max(self.eps, self.eps_min)

        return loss