import numpy as np
import socket
import pickle
import cv2

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import wandb

# Define Socket
HOST = '127.0.0.1'
timeout = 20

def done_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(5000)
    return_str = return_byt.decode() 

    return eval(return_str)

def reward_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(5000)
    return_str = return_byt.decode()
    if return_str == 'None':
        return_float = 0
    else:
        return_float = float(return_str) 

    return return_float

def obs_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(5000)
    # observation = return_byt.decode()
    observation = pickle.loads(return_byt)

    return observation

def send_ep_count_to_gh_client(socket, message):
    message_str = str(message)
    message_byt = message_str.encode()

    socket.listen()
    conn, _ = socket.accept()
    with conn:
        conn.send(message_byt)

def send_to_gh_client(socket, message):
    message_str = ''
    for item in message:
        listToStr = ' '.join(map(str, item))
        message_str = message_str + listToStr + '\n'

    message_byt = message_str.encode()
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        conn.send(message_byt)   

# Set device
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print(f'Used Device: {device}')

img_size = 256
def read_obs(fp):
    im = cv2.imread(fp, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_arr = np.array(im)
    im_arr = im_arr.reshape((3, img_size, img_size))
    im_arr = im_arr / 255.0
    state = torch.from_numpy(im_arr).type(torch.float32)

    return state

# Actor Critic Model Architecture 
class LSTMpolicy(nn.Module):
    def __init__(self, n_critic_layers, n_actor_layers, input_size, hidden_size, lin_size1):
        super(LSTMpolicy, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #critic
        self.critic_lstm = nn.LSTM(input_size, hidden_size, n_critic_layers, batch_first=True)
        self.critic_linear1 = nn.Linear(hidden_size, lin_size1)
        self.critic_linear2 = nn.Linear(lin_size1, 1)

        # actor
        self.actor_lstm1 = nn.LSTM(input_size, hidden_size, n_actor_layers, batch_first=True)
        self.actor_lstm2 = nn.LSTM(input_size, hidden_size, n_actor_layers, batch_first=True)
        self.actor_lstm3 = nn.LSTM(input_size, hidden_size, n_actor_layers, batch_first=True)
        self.actor_linear = nn.Linear(hidden_size, 17)

        self.relu = nn.ReLU()
    
    def forward(self, state, steps):
        state = Variable(torch.reshape(state, (1, steps, self.input_size)))

        # critic
        out, (hn, cn) = self.critic_lstm(state)
        hn = torch.squeeze(hn[-1, :, :])
        value = self.relu(hn)
        value = self.critic_linear1(value)
        value = self.relu(value)
        value = self.critic_linear2(value)

        # actor
        out1, (h1, c1) = self.actor_lstm1(state)
        out_l1 = torch.squeeze(h1[-1, :, :])
        out_l1 = self.relu(out_l1)
        out_l1 = self.actor_linear(out_l1)
        prob1 = F.softmax(out_l1, dim=-1)
        dist1 = Categorical(prob1)

        out2, (h2, c2) = self.actor_lstm2(state, (h1, c1))
        out_l2 = torch.squeeze(h2[-1, :, :])
        out_l2 = self.relu(out_l2)  
        out_l2 = self.actor_linear(out_l2)
        prob2 = F.softmax(out_l2, dim=-1)
        dist2 = Categorical(prob2)

        out3, (h3, c3) = self.actor_lstm3(state, (h2, c2))
        out_l3 = torch.squeeze(h3[-1, :, :])
        out_l3 = self.relu(out_l3)
        out_l3 = self.actor_linear(out_l3)
        prob3 = F.softmax(out_l3, dim=-1)
        dist3 = Categorical(prob3)
        
        return value, dist1, dist2, dist3 

def train():
    # hyperparameters
    hyperparameters = dict(n_steps = 10, # number of buildings per episode
                        n_episodes = 5000,
                        obs_len = 24,
                        gamma = 0.99,
                        beta = 0.001,
                        lr = 3e-4,
                        lr_decay = 0.1,
                        n_critic_layers = 2,
                        n_actor_layers = 2,
                        hidden_size = 256,
                        lin_size1 = 128,
                        lin_size2 = 64
                        )

    wandb.init(config=hyperparameters, entity='', project='') #Replace with your wandb entity & project
    # Save model inputs and hyperparameters
    config = wandb.config
    
    # Initialize DRL model
    actorcritic = LSTMpolicy(config.n_critic_layers, config.n_actor_layers, config.obs_len, config.hidden_size, 
                            config.lin_size1).to(device)
    ac_optimizer = optim.Adam(actorcritic.parameters(), lr=config.lr, weight_decay = 1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(ac_optimizer, mode='min', factor=config.lr_decay, patience=1000,
                                                    threshold=1e-5, threshold_mode='rel', cooldown=0,
                                                    min_lr=0, eps=1e-4, verbose=True)

    # Log gradients and model parameters wandb
    wandb.watch(actorcritic, log="all", log_freq=10)

    # Define action space
    param1_space = torch.from_numpy(np.linspace(start=0.1, stop=0.9, num=17))
    param2_space = torch.from_numpy(np.linspace(start=0.1, stop=0.9, num=17))
    param3_space = torch.from_numpy(np.linspace(start=0, stop=160, num=17))

    all_lengths = []
    average_lengths = []

    for episode in range(config.n_episodes):
        init_state = torch.zeros(1, config.n_steps, config.obs_len)
        param1L, param2L, param3L= [], [], []
        log_probs = []
        values = []
        rewards = []
        entropy = 0

        if episode == 0:
            print('\nStart Loop in GH Client...\n')

        for steps in range(config.n_steps):
            if steps == 0:
                state = init_state.to(device)
            # forward pass
            value, dist1, dist2, dist3 = actorcritic.forward(state, config.n_steps) 

            param1_idx = dist1.sample()
            param2_idx = dist2.sample()
            param3_idx = dist3.sample()

            param1 = param1_space[param1_idx]
            param2 = param2_space[param2_idx]
            param3 = param3_space[param3_idx] 

            log_prob = dist1.log_prob(param1_idx) + dist2.log_prob(param2_idx) + dist3.log_prob(param3_idx) # log(a*b) = log(a) + log(b)
            smoothed_entropy = dist1.entropy().mean() + dist2.entropy().mean() + dist3.entropy().mean()

            param1L.append(param1.item())
            param2L.append(param2.item())
            param3L.append(param3.item())

            action = [param1L, param2L, param3L]

            # Send action through socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8080))
                s.settimeout(timeout)
                send_to_gh_client(s, action)

            # Send episode count through socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8083))
                s.settimeout(timeout)
                send_ep_count_to_gh_client(s, episode)

            ######### In between GH script #########################################################

            # Receive observation from gh Client
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8084))
                s.settimeout(timeout)
                observation = obs_from_gh_client(s)

            # Receive Reward from gh Client
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8081))
                s.settimeout(timeout)
                reward = reward_from_gh_client(s)

            # Receive done from Client
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8082))
                s.settimeout(timeout)
                done = done_from_gh_client(s)

            # next state
            observation = torch.tensor(observation).to(device)
            next_state = state.clone()
            next_state[:, steps, :] = observation
            state = next_state

            rewards.append(torch.tensor(reward).unsqueeze(-1).to(device))
            values.append(value)
            log_probs.append(log_prob.unsqueeze(-1))
            entropy += smoothed_entropy.unsqueeze(-1)

            print(f"step {steps}, reward: {reward}, value: {value.item()}, log_prob: {log_prob}, entropy: {entropy.item()}")
            
            if done or steps == config.n_steps-1:
                Qval = 0
                all_lengths.append(steps + 1)
                average_lengths.append(np.mean(all_lengths))
                eps_reward = torch.sum(torch.cat(rewards)).item()
                print(f"episode {episode}, eps_reward: {eps_reward}, total length: {steps + 1}, average length: {average_lengths[-1]}")
                break
        
        # compute loss functions
        returns = []
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + config.gamma * Qval
            returns.insert(0, Qval)
            
        returns = torch.cat(returns)
        print(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean() 
        critic_loss = 0.5 * advantage.pow(2).mean() 
        ac_loss = actor_loss + critic_loss - config.beta * entropy

        # update actor critic
        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
        
        print(f"episode {episode}, actor_loss: {actor_loss.item()}, critic_loss: {critic_loss.item()}, ac_loss: {ac_loss.item()} \n")

        # Log metrics to visualize performance wandb
        wandb.log({
            'episode': episode, 
            'learning_rate': ac_optimizer.param_groups[0]['lr'], 
            'reward': eps_reward, 
            'actor_loss': actor_loss.item(), 
            'critic_loss': critic_loss.item(), 
            'ac_loss': ac_loss.item()
            })
        
        # update learning rate
        scheduler.step(critic_loss)
        print(f"current_lr: {ac_optimizer.param_groups[0]['lr']}")

        # save models
        # if eps_reward >= -4:
        #     torch.save(actorcritic.state_dict(), f'D:/RLinGUD/cutnfill_TDa2c_models/cutnfill-A2C_{episode}_{eps_reward}.pt')


if __name__ == "__main__":
    # Log in to W&B account
    wandb.login(key='') # place wandb key here!

    sweep = False
    if sweep:
        sweep_config = {
                'method': 'bayes', #grid, random, bayes
                'metric': {
                'name': 'reward',
                'goal': 'maximize'   
                },
                'parameters': {
                    'lr': {
                        'values':[1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
                    },
                    'n_critic_layers':{
                        'values':[1, 2, 4]
                    },
                    'n_actor_layers':{
                        'values':[1, 2]
                    },
                    'hidden_size':{
                        'values':[64, 128, 256, 512]
                    },
                    'lin_size1':{
                        'values':[64, 128, 256, 512]
                    }
                }
            }
        sweep_id = wandb.sweep(sweep_config, project='CutnFill_TDA2C')
        wandb.agent(sweep_id, train)

    else:
        train()
