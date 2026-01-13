import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from .model import weights_init_


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.copy_(target_param * (1.0 - tau) + param * tau)

def hard_update(target, source):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.copy_(param)


class QNetworkDiscrete(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, num_actions) # Outputs Q-value for each action
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class CategoricalPolicy(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, num_actions) # Outputs logits for each action
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits

    def sample(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class SACDiscreteAgent:
    def __init__(self, num_inputs, action_space, device, hidden_size, lr, gamma, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        num_actions = action_space.n

        self.critic = QNetworkDiscrete(num_inputs, num_actions, hidden_size).to(device)
        self.critic_target = QNetworkDiscrete(num_inputs, num_actions, hidden_size).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        hard_update(self.critic_target, self.critic)

        self.policy = CategoricalPolicy(num_inputs, num_actions, hidden_size).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, _ = self.policy.sample(state)
        else:
            # For eval, take the most likely action
            with torch.no_grad():
                logits = self.policy(state)
                action = logits.argmax(dim=-1)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # Calculate target Q-value
            next_logits = self.policy(next_state_batch)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            
            qf_next_target = self.critic_target(next_state_batch)
            # V(s') = sum(probs * (Q(s',a') - alpha * log_probs))
            v_next_target = (next_probs * (qf_next_target - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            next_q_value = reward_batch + mask_batch * self.gamma * v_next_target

        # --- Critic Update ---
        qf = self.critic(state_batch).gather(1, action_batch) # Get Q-value for the action taken
        qf_loss = F.mse_loss(qf, next_q_value)
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # --- Policy Update ---
        logits = self.policy(state_batch)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            qf_pi = self.critic(state_batch)

        policy_loss = (probs * (self.alpha * log_probs - qf_pi)).sum(dim=1).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # --- Soft Update Target Network ---
        soft_update(self.critic_target, self.critic, self.tau)