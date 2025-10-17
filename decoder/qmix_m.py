import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
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
        # sum of agent Q + state dim -> 1
        print("State dim: ", state_dim)
        print("Hidden dim: ", hidden_dim)
        self.fc1 = nn.Linear(state_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, agent_qs, state):
        q_mean = agent_qs.mean(dim=1, keepdim=True)
        x = torch.cat([state, q_mean], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QMIXAgent:
    def __init__(self, device="cuda"):
        self.device = device

        self.n_agents = cfg.n_agents
        self.obs_dim = cfg.obs_dim
        self.state_dim = cfg.state_dim
        self.act_dim = cfg.cluster_num

        self.agent_nets = nn.ModuleList([
            QAgentNet(self.obs_dim, self.act_dim) for _ in range(self.n_agents)
        ]).to(self.device)

        self.target_agent_nets = nn.ModuleList([
            QAgentNet(self.obs_dim, self.act_dim) for _ in range(self.n_agents)
        ]).to(self.device)

        for target_net, net in zip(self.target_agent_nets, self.agent_nets):
            target_net.load_state_dict(net.state_dict())

        # Mixer & Target Mixer
        self.mixer = QMixer(self.state_dim).to(self.device)
        self.target_mixer = QMixer(self.state_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Optimizer
        self.parameters = list(self.agent_nets.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=cfg.dqn_lr)

        self.replay_buffer = deque(maxlen=cfg.replay_size)

        # Epsilon 관련 설정
        self.eps_start = 1.0
        self.eps_min = 0.05
        self.eps = self.eps_start

        # Epoch 기반 decay 설정
        self.current_epoch = 0
        self.total_epochs = cfg.epochs if hasattr(cfg, 'epochs') else 100
        self.eps_decay_strategy = cfg.eps_decay_strategy if hasattr(cfg, 'eps_decay_strategy') else 'linear'

        # self.eps = 1.0
        # self.eps_min = 0.05
        # self.eps_decay = 1e-3
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
        if len(obs_list) < self.n_agents:
            obs_list = obs_list[:self.n_agents]
            actions = actions[:self.n_agents]
        if len(next_obs_list) < self.n_agents:
            next_obs_list = next_obs_list[:self.n_agents]
        self.replay_buffer.append((state, obs_list, actions, reward, next_state, next_obs_list))

    # def update(self):
    #     if len(self.replay_buffer) < self.batch_size:
    #         return None
    #
    #     batch = random.sample(self.replay_buffer, self.batch_size)
    #
    #     state_b, obs_b, actions_b, rewards_b, next_state_b, next_obs_b = zip(*batch)
    #     # state_b: (batch_size, state_dim)
    #     # obs_b: (batch_size, n_agents, obs_dim)
    #     # actions_b: (batch_size, n_agents)
    #     # rewards_b: (batch_size,)
    #     # next_state_b: (batch_size, state_dim)
    #     # next_obs_b: (batch_size, n_agents, obs_dim)
    #     state_t = torch.FloatTensor(state_b).to(self.device)
    #     next_state_t = torch.FloatTensor(next_state_b).to(self.device)
    #     rewards_t = torch.FloatTensor(rewards_b).to(self.device)
    #
    #     # obs_t = torch.FloatTensor(obs_b).to(self.device)
    #     # next_obs_t = torch.FloatTensor(next_obs_b).to(self.device)
    #     # actions_t = torch.LongTensor(actions_b).to(self.device)
    #
    #     # Current Q
    #     n_agents_in_batch = self.n_agents
    #
    #     # obs_t: (B, n_agents_in_batch, obs_dim)
    #     # actions_t: (B, n_agents_in_batch)
    #     obs_t = []
    #     actions_t = []
    #     for b_idx in range(self.batch_size):
    #         active_agents = len(obs_b[b_idx])
    #         # print("Active agents: ", active_agents)
    #         # print("Observation of a batch", np.array(obs_b[b_idx]).shape)
    #
    #         obs_pad = np.zeros((n_agents_in_batch, self.obs_dim), dtype=np.float32)
    #         act_pad = np.zeros((n_agents_in_batch,), dtype=np.int64)
    #         obs_pad[:active_agents] = np.array(obs_b[b_idx])
    #         act_pad[:active_agents] = np.array(actions_b[b_idx])
    #
    #         obs_t.append(obs_pad)
    #         actions_t.append(act_pad)
    #
    #     obs_t = torch.FloatTensor(obs_t).to(self.device)            # (B, n_agents_in_batch, obs_dim)
    #     actions_t = torch.LongTensor(actions_t).to(self.device)     # (B, n_agents_in_batch)
    #
    #     # next_obs_t
    #     next_obs_t = []
    #     for b_idx in range(self.batch_size):
    #         active_agents = len(next_obs_b[b_idx])
    #         obs_pad = np.zeros((n_agents_in_batch, self.obs_dim), dtype=np.float32)
    #         obs_pad[:active_agents] = np.array(next_obs_b[b_idx])
    #         next_obs_t.append(obs_pad)
    #
    #     next_obs_t = torch.FloatTensor(next_obs_t).to(self.device)
    #
    #     # Current Q
    #     # (B, n_agents_in_batch, act_dim)
    #     all_q = []
    #     for i in range(n_agents_in_batch):
    #         q_i = self.agent_nets[i](obs_t[:, i, :])            # (B, act_dim)
    #         all_q.append(q_i.unsqueeze(1))                      # (B, 1, act_dim)
    #     all_q = torch.cat(all_q, dim=1)
    #
    #     # Selected Q(s_i, a_i)
    #     chosen_q = []
    #     for i in range(n_agents_in_batch):
    #         q_i = all_q[:, i, :]                                # (B, act_dim)
    #         a_i = actions_t[:, i].unsqueeze(1)                  # (B, 1)
    #         q_i_chosen = q_i.gather(1, a_i).squeeze(1)      # (B,)
    #         chosen_q.append(q_i_chosen.unsqueeze(1))
    #     chosen_q = torch.cat(chosen_q, dim=1)                   # (B, n_agents_in_batch)
    #
    #     # Calculate Joint Q from QMixer
    #     print("chosen q shape: ", chosen_q.shape)
    #     print("state t shape: ", state_t.shape)
    #     q_total = self.mixer(chosen_q, state_t).squeeze(-1)
    #     print("q total shape: ", q_total.shape)
    #
    #     # Target Q
    #     with torch.no_grad():
    #         # target q
    #         target_q_list = []
    #         for i in range(n_agents_in_batch):
    #             q_i_next = self.target_agent_nets[i](next_obs_t[:, i, :])
    #             q_i_max = q_i_next.max(dim=1)[0].unsqueeze(1)
    #             target_q_list.append(q_i_max)
    #
    #         target_q = torch.cat(target_q_list, dim=1)          # (B, n_agents_in_batch)
    #         q_total_next = self.target_mixer(target_q, next_state_t).squeeze(-1)
    #
    #         target_value = rewards_t + self.gamma * q_total_next
    #
    #     loss = F.mse_loss(q_total, target_value.detach())
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     self.train_step += 1
    #     if self.train_step % self.update_interval == 0:
    #         for target_net, net in zip(self.target_agent_nets, self.agent_nets):
    #             target_net.load_state_dict(net.state_dict())
    #         self.target_mixer.load_state_dict(self.mixer.state_dict())
    #
    #     # Epsilon decay
    #     # if self.eps > self.eps_min:
    #     #     self.eps -= self.eps_decay
    #     #     self.eps = max(self.eps, self.eps_min)
    #
    #     return loss

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_b, obs_b, actions_b, rewards_b, next_state_b, next_obs_b = zip(*batch)

        # ========== 디버깅 코드 추가 ==========
        # 1. 각 state의 shape 확인
        print(f"[DEBUG] Batch size: {self.batch_size}")
        print(f"[DEBUG] State shapes in batch:")
        for i, state in enumerate(state_b):
            if isinstance(state, np.ndarray):
                print(f"  State {i}: shape={state.shape}, dtype={state.dtype}")
            else:
                print(f"  State {i}: type={type(state)}, len={len(state) if hasattr(state, '__len__') else 'N/A'}")

        # 2. 모든 state가 같은 shape인지 확인
        state_shapes = [np.array(s).shape for s in state_b]
        unique_shapes = list(set(state_shapes))
        if len(unique_shapes) > 1:
            print(f"[WARNING] Inconsistent state shapes found: {unique_shapes}")
            print(f"[WARNING] Shape distribution:")
            for shape in unique_shapes:
                count = state_shapes.count(shape)
                print(f"  Shape {shape}: {count} occurrences")

        # 3. 차원 불일치 처리 방법 1: 패딩
        try:
            # 가장 큰 차원 찾기
            max_state_dim = max(len(s) if hasattr(s, '__len__') else 1 for s in state_b)

            # 패딩 적용
            padded_states = []
            for state in state_b:
                state_array = np.array(state).flatten()  # 1D로 변환
                if len(state_array) < max_state_dim:
                    # 0으로 패딩
                    padded = np.zeros(max_state_dim)
                    padded[:len(state_array)] = state_array
                    padded_states.append(padded)
                else:
                    padded_states.append(state_array[:max_state_dim])  # 잘라내기

            state_t = torch.FloatTensor(padded_states).to(self.device)

        except Exception as e:
            print(f"[ERROR] Failed to create state tensor with padding: {e}")

            # 차원 불일치 처리 방법 2: 각 state를 개별적으로 처리
            try:
                # state를 리스트로 유지하고 나중에 처리
                state_list = [torch.FloatTensor(np.array(s).flatten()).to(self.device) for s in state_b]
                # 가장 작은 차원으로 통일
                min_dim = min(s.shape[0] for s in state_list)
                state_t = torch.stack([s[:min_dim] for s in state_list])

            except Exception as e2:
                print(f"[ERROR] Alternative state tensor creation also failed: {e2}")
                # 문제가 있는 배치 건너뛰기
                return None

        # next_state도 같은 방식으로 처리
        try:
            max_next_state_dim = max(len(s) if hasattr(s, '__len__') else 1 for s in next_state_b)
            padded_next_states = []
            for state in next_state_b:
                state_array = np.array(state).flatten()
                if len(state_array) < max_next_state_dim:
                    padded = np.zeros(max_next_state_dim)
                    padded[:len(state_array)] = state_array
                    padded_next_states.append(padded)
                else:
                    padded_next_states.append(state_array[:max_next_state_dim])

            next_state_t = torch.FloatTensor(padded_next_states).to(self.device)

        except Exception as e:
            print(f"[ERROR] Failed to create next_state tensor: {e}")
            return None

        rewards_t = torch.FloatTensor(rewards_b).to(self.device)

        # ========== 나머지 코드는 동일 ==========

        # Current Q
        n_agents_in_batch = self.n_agents

        # obs_t: (B, n_agents_in_batch, obs_dim)
        # actions_t: (B, n_agents_in_batch)
        obs_t = []
        actions_t = []
        for b_idx in range(self.batch_size):
            active_agents = len(obs_b[b_idx])

            obs_pad = np.zeros((n_agents_in_batch, self.obs_dim), dtype=np.float32)
            act_pad = np.zeros((n_agents_in_batch,), dtype=np.int64)
            obs_pad[:active_agents] = np.array(obs_b[b_idx])
            act_pad[:active_agents] = np.array(actions_b[b_idx])

            obs_t.append(obs_pad)
            actions_t.append(act_pad)

        obs_t = torch.FloatTensor(obs_t).to(self.device)
        actions_t = torch.LongTensor(actions_t).to(self.device)

        # next_obs_t
        next_obs_t = []
        for b_idx in range(self.batch_size):
            active_agents = len(next_obs_b[b_idx])
            obs_pad = np.zeros((n_agents_in_batch, self.obs_dim), dtype=np.float32)
            obs_pad[:active_agents] = np.array(next_obs_b[b_idx])
            next_obs_t.append(obs_pad)

        next_obs_t = torch.FloatTensor(next_obs_t).to(self.device)

        # Current Q
        all_q = []
        for i in range(n_agents_in_batch):
            q_i = self.agent_nets[i](obs_t[:, i, :])
            all_q.append(q_i.unsqueeze(1))
        all_q = torch.cat(all_q, dim=1)

        # Selected Q(s_i, a_i)
        chosen_q = []
        for i in range(n_agents_in_batch):
            q_i = all_q[:, i, :]
            a_i = actions_t[:, i].unsqueeze(1)
            q_i_chosen = q_i.gather(1, a_i).squeeze(1)
            chosen_q.append(q_i_chosen.unsqueeze(1))
        chosen_q = torch.cat(chosen_q, dim=1)

        # Calculate Joint Q from QMixer
        print("chosen q shape: ", chosen_q.shape)
        print("state t shape: ", state_t.shape)
        q_total = self.mixer(chosen_q, state_t).squeeze(-1)
        print("q total shape: ", q_total.shape)

        # Target Q
        with torch.no_grad():
            target_q_list = []
            for i in range(n_agents_in_batch):
                q_i_next = self.target_agent_nets[i](next_obs_t[:, i, :])
                q_i_max = q_i_next.max(dim=1)[0].unsqueeze(1)
                target_q_list.append(q_i_max)

            target_q = torch.cat(target_q_list, dim=1)
            q_total_next = self.target_mixer(target_q, next_state_t).squeeze(-1)

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

        return loss

    # 추가 헬퍼 함수: Replay Buffer에 데이터 추가 시 검증
    def add_to_replay_buffer_with_validation(self, state, obs, actions, reward, next_state, next_obs):
        """
        Replay buffer에 데이터를 추가하기 전에 차원을 검증하고 정규화합니다.
        """
        # state와 next_state를 numpy array로 변환하고 shape 확인
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()

        # 예상 차원과 일치하는지 확인
        if hasattr(self, 'state_dim'):
            if len(state) != self.state_dim:
                print(f"[WARNING] State dimension mismatch: expected {self.state_dim}, got {len(state)}")
                # 패딩 또는 잘라내기
                if len(state) < self.state_dim:
                    padded_state = np.zeros(self.state_dim)
                    padded_state[:len(state)] = state
                    state = padded_state
                else:
                    state = state[:self.state_dim]

            if len(next_state) != self.state_dim:
                if len(next_state) < self.state_dim:
                    padded_next_state = np.zeros(self.state_dim)
                    padded_next_state[:len(next_state)] = next_state
                    next_state = padded_next_state
                else:
                    next_state = next_state[:self.state_dim]

        # Replay buffer에 추가
        self.replay_buffer.append((state, obs, actions, reward, next_state, next_obs))

        # 버퍼 크기 제한
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)