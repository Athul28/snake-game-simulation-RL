# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Original DQN training step (keeps using max over next state's q-values).
        """

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def train_step_sarsa(self, states, actions, rewards, next_states, next_actions, dones):
        """
        SARSA update using NN function approximator.
        Expects:
          states, actions, rewards, next_states, next_actions, dones
        where actions/next_actions are one-hot lists or tensors.
        """

        state = torch.tensor(states, dtype=torch.float)
        next_state = torch.tensor(next_states, dtype=torch.float)
        action = torch.tensor(actions, dtype=torch.long)
        next_action = torch.tensor(next_actions, dtype=torch.long)
        reward = torch.tensor(rewards, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            next_action = torch.unsqueeze(next_action, 0)
            reward = torch.unsqueeze(reward, 0)
            dones = (dones, )

        # Predicted Q for current states and for next states
        pred = self.model(state)            # shape: (batch, num_actions)
        pred_next = self.model(next_state) # shape: (batch, num_actions)

        target = pred.clone()

        for idx in range(len(dones)):
            # extract index of the action taken (assumes one-hot / single-hot)
            act_idx = torch.argmax(action[idx]).item()
            next_act_idx = torch.argmax(next_action[idx]).item()

            Q_new = reward[idx]
            if not dones[idx]:
                # SARSA uses Q(s', a') (on-policy) rather than max_a' Q(s', a')
                Q_new = reward[idx] + self.gamma * pred_next[idx][next_act_idx].detach()

            target[idx][act_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()