import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Enable anomaly detection (optional)
torch.autograd.set_detect_anomaly(True)

# ===============================
# 1. Build the simulation environment (use new_step_api=True to eliminate warnings)
# ===============================
env = gym.make('CartPole-v1', new_step_api=True)

# ===============================
# 2. Data Preprocessing
# Normalize the observation to the [0, 1] range
# ===============================
def normalize_observation(obs, obs_min, obs_max):
    return (obs - obs_min) / (obs_max - obs_min)

obs_min = np.array([-4.8, -5.0, -0.418, -5.0])
obs_max = np.array([4.8, 5.0, 0.418, 5.0])

# ===============================
# 3. Perception Module (Autoencoder)
# Transform the 4-dimensional raw state into a 2-dimensional latent representation
# ===============================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),  # non in-place version
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
# 4. Memory Module (LSTM)
# Used to process continuous latent state sequences and capture temporal dependencies
# ===============================
class SimpleMemory(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, num_layers=1):
        super(SimpleMemory, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

# ===============================
# 5. Prediction Module
# Used to predict the next latent state
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
# 6. Decision Module
# Use the prediction module to evaluate candidate actions and select the one with the highest utility
# ===============================
def choose_action(state_latent, predictor, device):
    actions = [0, 1]
    state_tensor = state_latent.unsqueeze(0)  # shape: [1, latent_dim]
    action_values = []
    for a in actions:
        action_tensor = torch.tensor([[float(a)]], device=device, dtype=torch.float32)
        pred_next = predictor(state_tensor, action_tensor)
        value = pred_next[0, 0].item()  # simple example strategy
        action_values.append(value)
    best_action = actions[np.argmax(action_values)]
    return best_action

# ===============================
# Initialize modules and optimizers
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = AutoEncoder(input_dim=4, latent_dim=2).to(device)
memory_module = SimpleMemory(input_dim=2, hidden_dim=4).to(device)
predictor = Predictor(state_dim=2, action_dim=1, hidden_dim=8).to(device)

ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
pred_optimizer = optim.Adam(predictor.parameters(), lr=0.001)

# ===============================
# 7. Closed-loop System Training and Interaction
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
        # Data preprocessing
        obs = np.array(obs, dtype=np.float32)
        norm_obs = normalize_observation(obs, obs_min, obs_max)
        obs_tensor = torch.tensor(norm_obs, device=device, dtype=torch.float32).unsqueeze(0)

        # Perception module: obtain the latent representation
        _, latent = autoencoder(obs_tensor)
        latent = latent.squeeze(0)
        latent_sequence.append(latent)

        # Action selection
        if len(latent_sequence) < sequence_length:
            action = env.action_space.sample()
        else:
            seq_tensor = torch.stack(latent_sequence[-sequence_length:]).unsqueeze(0)
            memory_out, hidden_state = memory_module(seq_tensor, hidden_state)
            if hidden_state is not None:
                hidden_state = tuple([h.detach() for h in hidden_state])
            context_state = memory_out[:, -1, :]
            action = choose_action(latent, predictor, device)

        # Execute action (new API returns 5 values)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Autoencoder update: reconstruct the current observation
        recon, _ = autoencoder(obs_tensor)
        ae_loss = nn.MSELoss()(recon, obs_tensor)
        ae_optimizer.zero_grad()
        ae_loss.backward()
        ae_optimizer.step()

        # Predictor module update: predict the next latent
        if len(latent_sequence) >= sequence_length:
            action_tensor = torch.tensor([[float(action)]], device=device, dtype=torch.float32)
            # Detach the gradient from the autoencoder output
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
