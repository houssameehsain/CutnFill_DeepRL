import numpy as np
import socket
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

def fp_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(5000)
    fp = return_byt.decode()

    return fp

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
def enc_block(in_c, out_c, BN=True):
    if BN:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return conv
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        return conv
class GRUpolicy(nn.Module):
    def __init__(self, n_gru_layers, hidden_size, lin_size1, lin_size2, enc_size1, enc_size2, enc_size3):
        super(GRUpolicy, self).__init__()

        #critic
        self.critic_enc1 = enc_block(3, enc_size1, BN=False)
        self.critic_enc2 = enc_block(enc_size1, enc_size2, BN=True)
        self.critic_enc3 = enc_block(enc_size2, enc_size3, BN=True)
        self.critic_enc4 = enc_block(enc_size3, 128, BN=True)

        self.critic_linear1 = nn.Linear(512, lin_size1)
        self.critic_linear2 = nn.Linear(lin_size1, lin_size2)
        self.critic_linear3 = nn.Linear(lin_size2, 1)

        # actor
        self.gru1 = nn.GRU(4, hidden_size, n_gru_layers, batch_first=True)
        self.gru2 = nn.GRU(4, hidden_size, n_gru_layers, batch_first=True)
        self.gru3 = nn.GRU(4, hidden_size, n_gru_layers, batch_first=True)
        self.actor_linear = nn.Linear(hidden_size, 17)
    
    def forward(self, state):
        state = Variable(state.unsqueeze(0))

        # critic
        enc = self.critic_enc1(state)
        enc = self.critic_enc2(enc)
        enc = self.critic_enc3(enc)
        enc = self.critic_enc4(enc)

        value = F.relu(self.critic_linear1(torch.flatten(enc)))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)

        # actor
        seq = torch.reshape(enc, (1, 128, 4))

        out1, h_1 = self.gru1(seq)
        out_s1 = torch.squeeze(out1[:, -1, :])
        out_l1 = self.actor_linear(out_s1)
        prob1 = F.softmax(out_l1, dim=-1)
        dist1 = Categorical(prob1)

        out2, h_2 = self.gru2(seq, h_1)  
        out_s2 = torch.squeeze(out2[:, -1, :])
        out_l2 = self.actor_linear(out_s2)
        prob2 = F.softmax(out_l2, dim=-1)
        dist2 = Categorical(prob2)

        out3, _ = self.gru3(seq, h_2)
        out_s3 = torch.squeeze(out3[:, -1, :])
        out_l3 = self.actor_linear(out_s3)
        prob3 = F.softmax(out_l3, dim=-1)
        dist3 = Categorical(prob3)
        
        return value, dist1, dist2, dist3 

def train():
    # hyperparameters
    hyperparameters = dict(n_steps = 1, # number of buildings per episode
                        n_episodes = 100,
                        gamma = 0.99,
                        beta = 0.001,
                        lr = 3e-3,
                        lr_decay = 0.1,
                        n_gru_layers = 1,
                        hidden_size = 256,
                        lin_size1 = 128,
                        lin_size2 = 64,
                        enc_size1 = 256,
                        enc_size2 = 256,
                        enc_size3 = 32)

    wandb.init(config=hyperparameters, entity='', project='') #Replace with your wandb entity & project
    # Save model inputs and hyperparameters
    config = wandb.config
    
    # Initialize DRL model
    actorcritic = GRUpolicy(config.n_gru_layers, config.hidden_size, config.lin_size1, config.lin_size2, 
                            config.enc_size1, config.enc_size2, config.enc_size3).to(device)
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
        fps = []
        param1L, param2L, param3L= [], [], []
        log_probs = []
        values = []
        rewards = []
        entropy = 0

        if episode == 0:
            print('\nStart Loop in GH Client...\n')

        for steps in range(config.n_steps):
            if steps == 0:
                fp = 'D:/RLinGUD/cutnfill_obs/observation_init.png' # replace with intial state image file path
            else:
                fp = fps[-1]
            
            # Get observation from Memory
            state = read_obs(fp).to(device)
            value, dist1, dist2, dist3 = actorcritic.forward(state) 

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

            # Receive observation file path from gh Client
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, 8084))
                s.settimeout(timeout)
                fp = fp_from_gh_client(s)

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

            fps.append(fp) # next state
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
            
        returns = torch.cat(returns).detach()
        print(returns)
        values = torch.cat(values)
        print(values)
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
                    'n_gru_layers':{
                        'values':[1, 2]
                    },
                    'hidden_size':{
                        'values':[64, 128, 256, 512]
                    },
                    'lin_size1':{
                        'values':[64, 128, 256, 512]
                    },
                    'lin_size2':{
                        'values':[64, 128, 256, 512]
                    },
                    'enc_size1':{
                        'values':[64, 128, 256, 512]
                    },
                    'enc_size2':{
                        'values':[64, 128, 256, 512]
                    },
                    'enc_size3':{
                        'values':[64, 128, 256, 512]
                    }
                }
            }
        sweep_id = wandb.sweep(sweep_config, project='CutnFill_TDA2C')
        wandb.agent(sweep_id, train)

    else:
        train()