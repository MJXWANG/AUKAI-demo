import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 开启异常检测（可选）
torch.autograd.set_detect_anomaly(True)

# ===============================
# 1. 搭建仿真环境（使用 new_step_api=True 消除警告）
# ===============================
env = gym.make('CartPole-v1', new_step_api=True)

# ===============================
# 2. 数据预处理
# 将观测归一化到 [0, 1] 区间
# ===============================
def normalize_observation(obs, obs_min, obs_max):
    return (obs - obs_min) / (obs_max - obs_min)

obs_min = np.array([-4.8, -5.0, -0.418, -5.0])
obs_max = np.array([4.8, 5.0, 0.418, 5.0])

# ===============================
# 3. 感知模块（自动编码器）
# 将 4 维原始状态转换为 2 维低维表示（latent state）
# ===============================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),  # 非 in-place 版本
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# ===============================
# 4. 记忆模块（LSTM）
# 用于处理连续低维状态序列，捕捉时间依赖性
# ===============================
class SimpleMemory(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, num_layers=1):
        super(SimpleMemory, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

# ===============================
# 5. 预测模块
# 用于预测下一个低维状态（latent state）
# ===============================
class Predictor(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=8):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        next_state_pred = self.fc(x)
        return next_state_pred

# ===============================
# 6. 决策模块
# 采用预测模块对候选动作进行评估，选择效用最高的动作
# ===============================
def choose_action(state_latent, predictor, device):
    actions = [0, 1]
    state_tensor = state_latent.unsqueeze(0)  # shape: [1, latent_dim]
    action_values = []
    for a in actions:
        action_tensor = torch.tensor([[float(a)]], device=device, dtype=torch.float32)
        pred_next = predictor(state_tensor, action_tensor)
        value = pred_next[0, 0].item()  # 简单示例策略
        action_values.append(value)
    best_action = actions[np.argmax(action_values)]
    return best_action

# ===============================
# 初始化各模块及优化器
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = AutoEncoder(input_dim=4, latent_dim=2).to(device)
memory_module = SimpleMemory(input_dim=2, hidden_dim=4).to(device)
predictor = Predictor(state_dim=2, action_dim=1, hidden_dim=8).to(device)

ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
pred_optimizer = optim.Adam(predictor.parameters(), lr=0.001)

# ===============================
# 7. 闭环系统训练与交互
# ===============================
num_episodes = 10
max_steps = 200
sequence_length = 5

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    latent_sequence = []
    hidden_state = None

    for step in range(max_steps):
        # 数据预处理
        obs = np.array(obs, dtype=np.float32)
        norm_obs = normalize_observation(obs, obs_min, obs_max)
        obs_tensor = torch.tensor(norm_obs, device=device, dtype=torch.float32).unsqueeze(0)

        # 感知模块：获得低维表示
        _, latent = autoencoder(obs_tensor)
        latent = latent.squeeze(0)
        latent_sequence.append(latent)

        # 动作选择
        if len(latent_sequence) < sequence_length:
            action = env.action_space.sample()
        else:
            seq_tensor = torch.stack(latent_sequence[-sequence_length:]).unsqueeze(0)
            memory_out, hidden_state = memory_module(seq_tensor, hidden_state)
            if hidden_state is not None:
                hidden_state = tuple([h.detach() for h in hidden_state])
            context_state = memory_out[:, -1, :]
            action = choose_action(latent, predictor, device)

        # 执行动作（新 API 返回 5 个值）
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # 自动编码器更新：重构当前观察
        recon, _ = autoencoder(obs_tensor)
        ae_loss = nn.MSELoss()(recon, obs_tensor)
        ae_optimizer.zero_grad()
        ae_loss.backward()
        ae_optimizer.step()

        # 预测模块更新：预测下一个 latent
        if len(latent_sequence) >= sequence_length:
            action_tensor = torch.tensor([[float(action)]], device=device, dtype=torch.float32)
            # 这里断开自动编码器输出的梯度传递
            pred_next = predictor(latent.detach().unsqueeze(0), action_tensor)
            with torch.no_grad():
                norm_next = normalize_observation(next_obs, obs_min, obs_max)
                next_obs_tensor = torch.tensor(norm_next, device=device, dtype=torch.float32).unsqueeze(0)
                _, next_latent = autoencoder(next_obs_tensor)
                next_latent = next_latent.clone().detach()
            pred_loss = nn.MSELoss()(pred_next, next_latent)
            pred_optimizer.zero_grad()
            pred_loss.backward()
            pred_optimizer.step()

        obs = next_obs
        if done:
            break

    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()
