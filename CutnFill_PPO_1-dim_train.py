import numpy as np
import socket
import pickle

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import wandb   

# Set device
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print(f'Used Device: {device}')

# hyperparameters
hyperparameters = dict(n_steps = 500,
                    max_frames = 500000,
                    mini_batch_size = 50,
                    ppo_epochs = 10,
                    max_ep_steps = 10,
                    obs_len = 24,
                    gamma = 0.99,
                    tau = 0.95,
                    beta = 0.001,
                    lr = 3e-4,
                    lr_schedule = False,
                    lr_decay = 0.1,
                    hidden1 = 256,
                    hidden2 = 256
                    )

# Log in to W&B account
wandb.login(key='')

wandb.init(config=hyperparameters, entity='', project='CutnFill_PPO')
# Save model inputs and hyperparameters
config = wandb.config

# Actor Critic Model Architecture 
class policy(nn.Module):
    def __init__(self, environment, input_size, steps, hidden1, hidden2):
        super(actorcritic, self).__init__()
        self.input_size = input_size
        self.steps = steps
        self.out_size = environment.action_space.size()[0]
        self.input = self.input_size*self.steps

        #critic
        self.critic_linear1 = nn.Linear(self.input, hidden1)
        self.critic_linear2 = nn.Linear(hidden1, hidden2)
        self.critic_linear3 = nn.Linear(hidden2, 1)

        # actor
        self.actor_linear1 = nn.Linear(self.input, hidden1)
        self.actor_linear2 = nn.Linear(hidden1, hidden2)
        self.actor_linear3 = nn.Linear(hidden2, self.out_size)

        self.relu = nn.ReLU()
    
    def forward(self, state):
        state = Variable(torch.flatten(state, start_dim=1))

        # critic
        value = self.critic_linear1(state)
        value = self.relu(value)
        value = self.critic_linear2(value)
        value = self.relu(value)
        value = self.critic_linear3(value)

        # actor
        out = self.actor_linear1(state)
        out = self.relu(out)
        out = self.actor_linear2(out)
        out = self.relu(out)
        out = self.actor_linear3(out)
        prob = F.softmax(out, dim=-1)
        dist = Categorical(prob)
        
        return value, dist

def compute_gae(next_value, rewards, masks, values, gamma, tau):
            values = values + [next_value]
            gae = 0
            returns = []
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
                gae = delta + gamma * tau * masks[t] * gae
                returns.insert(0, gae + values[t])
            return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for epoch in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):

            value, dist = actorcritic.forward(state)

            entropy = dist.entropy() 
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (return_ - value).pow(2).mean()

            ac_loss = critic_loss + actor_loss - 0.001 * entropy
            ac_loss = ac_loss.mean()

            ac_optimizer.zero_grad()
            ac_loss.backward()
            ac_optimizer.step()

            # update learning rate
            if config.lr_schedule:
                scheduler.step(ac_loss)
                print(f"current_lr: {ac_optimizer.param_groups[0]['lr']}")

            print(f"ppo_epoch: {epoch}, actor_loss: {actor_loss.item()}, critic_loss: {critic_loss.item()}, ac_loss: {ac_loss.item()} \n")

            # Log metrics to visualize performance wandb
            wandb.log({ 
                'learning_rate': ac_optimizer.param_groups[0]['lr'],  
                'actor_loss': actor_loss.item(), 
                'critic_loss': critic_loss.item(), 
                'ac_loss': ac_loss.item()
                })

def done_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(5000)
    done = pickle.loads(return_byt) 

    if done != True and done != False:
        done = True

    return done

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

    observation = pickle.loads(return_byt)

    return observation

def send_render_to_gh_client(socket, message):
    render_byt = pickle.dumps(message)

    socket.listen()
    conn, _ = socket.accept()
    with conn:
        conn.send(render_byt)

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

class environment():
    def __init__(self, max_ep_steps, obs_len):

        self.HOST = '127.0.0.1'
        self.timeout = 20
        
        self.threshold_reward = -3.5

        self.obs_len = obs_len
        self.max_ep_steps = max_ep_steps
        self.param1, self.param2, self.param3 = [], [], []
        self.test_param1, self.test_param2, self.test_param3 = [], [], []

        # Define action space
        self.param1_space = torch.from_numpy(np.linspace(start=0.1, stop=0.9, num=17))
        self.param2_space = torch.from_numpy(np.linspace(start=0.1, stop=0.9, num=17))
        self.param3_space = torch.from_numpy(np.linspace(start=0, stop=160, num=17))

        self.action_space = torch.from_numpy(np.array(np.meshgrid(self.param1_space, self.param2_space, self.param3_space)).T.reshape(-1,3))

    def reset(self, test=False):
        if not test:
            self.param1, self.param2, self.param3 = [], [], []
            init_state = torch.zeros(1, self.max_ep_steps, self.obs_len).to(device)
        else:
            self.test_param1, self.test_param2, self.test_param3 = [], [], []
            init_state = torch.zeros(1, self.max_ep_steps, self.obs_len).to(device)
        
        return init_state

    def step(self, action, render=False):
        # Send action through socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.HOST, 8080))
            s.settimeout(self.timeout)
            send_to_gh_client(s, action)
        
        # Send render bool through socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.HOST, 8083))
            s.settimeout(self.timeout)
            send_render_to_gh_client(s, render)

        ######### In between GH script ##############################

        # Receive observation from gh Client
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.HOST, 8084))
            s.settimeout(self.timeout)
            observation = obs_from_gh_client(s)

        # Recieve Reward from gh Client
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.HOST, 8081))
            s.settimeout(self.timeout)
            reward = reward_from_gh_client(s)

        # Recieve done from Client
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.HOST, 8082))
            s.settimeout(self.timeout)
            done = done_from_gh_client(s)
        
        return observation, reward, done

    def test(self, render=True):
        test_state = self.reset(test=True)
        done = False
        eps_reward = 0
        ep_length = 0
        while not done:
            _, dist = actorcritic(test_state)

            action_idx = dist.sample()
            action_idx = torch.tensor([action_idx]).to(device)
            action = self.action_space[action_idx.item(), :]

            self.test_param1.append(action[0].item())
            self.test_param2.append(action[1].item())
            self.test_param3.append(action[2].item())

            gh_action = [self.test_param1, self.test_param2, self.test_param3]

            observation, reward, _ = self.step(gh_action, render=render)

            # next state
            observation = torch.tensor(observation).to(device)
            test_next_state = test_state.clone()
            test_next_state[:, ep_length, :] = observation
            test_state = test_next_state

            ep_length += 1
            eps_reward += reward

            # terminal state when max building count is reached
            if len(self.test_param1) == self.max_ep_steps:
                done = True
        
        return eps_reward, ep_length 


# environment instance
env = environment(config.max_ep_steps, config.obs_len)

# Initialize DRL model
actorcritic = policy(env, config.obs_len, config.max_ep_steps, config.hidden1, config.hidden2).to(device)
ac_optimizer = optim.Adam(actorcritic.parameters(), lr=config.lr, weight_decay = 1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(ac_optimizer, mode='min', factor=config.lr_decay, patience=1000,
                                                threshold=1e-4, threshold_mode='rel', cooldown=0,
                                                min_lr=0, eps=1e-4, verbose=True)

# Log gradients and model parameters wandb
wandb.watch(actorcritic, log="all", log_freq=10)


def train():
    state = env.reset()
    ep_lengths = []

    frame_idx = 0
    test_count = 0
    enable_early_stop = True
    early_stop = False
    while frame_idx < config.max_frames and not early_stop:

        states = []
        actions_idx = []
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        if frame_idx == 0:
            print('\nStart Loop in GH Client...\n')

        ep_len = 0
        for _ in range(config.n_steps):

            # Forward pass
            value, dist = actorcritic.forward(state) 

            action_idx = dist.sample()
            action_idx = torch.tensor([action_idx]).to(device)
            action = env.action_space[action_idx.item(), :] 

            env.param1.append(action[0].item())
            env.param2.append(action[1].item())
            env.param3.append(action[2].item())

            gh_action = [env.param1, env.param2, env.param3]

            observation, reward, _ = env.step(gh_action)

            # next state
            observation = torch.tensor(observation).to(device)
            next_state = state.clone()
            next_state[:, ep_len, :] = observation
            
            log_prob = dist.log_prob(action_idx) 
            entropy = dist.entropy()
            entropy += entropy

            # terminal state when max building count is reached
            if len(env.param1) == config.max_ep_steps:
                done = True
            else:
                done = False
            
            rewards.append(torch.tensor(reward).unsqueeze(-1).to(device))
            masks.append(torch.tensor(1 - done).unsqueeze(-1).to(device))
            values.append(value)
            log_probs.append(log_prob.unsqueeze(-1))

            states.append(state)
            actions_idx.append(action_idx)

            if not done:
                state = next_state
                ep_len += 1
            else:
                state = env.reset()
                ep_len = 0

            frame_idx += 1
            
            if frame_idx % 2500 == 0:
                eps_rewards = []
                for i in range(10):
                    eps_reward, ep_length = env.test()
                    eps_rewards.append(eps_reward)
                    ep_lengths.append(ep_length)
                    print(f"frame_idx: {frame_idx}, episode: {i}, eps_reward: {eps_reward}, total length: {ep_length}")

                mean_reward = np.mean(eps_rewards)
                mean_ep_length = np.mean(ep_lengths)  
                print(f"frame_idx: {frame_idx}, mean_reward: {mean_reward}, mean_ep_length: {mean_ep_length}")

                test_count += 2500
                # Log metrics to visualize performance wandb
                wandb.log({
                    'reward': mean_reward,
                    'episode_length': mean_ep_length, 
                    'frames': test_count
                })

                if enable_early_stop and mean_reward > env.threshold_reward:
                    early_stop = True
                
        next_value, _ = actorcritic.forward(next_state)
        
        # compute generalized advantage estimate GAE
        returns = compute_gae(next_value, rewards, masks, values, config.gamma, config.tau)

        returns = torch.cat(returns).detach()
        values = torch.cat(values).detach()
        log_probs = torch.cat(log_probs).detach()
        states = torch.cat(states)
        actions_idx = torch.cat(actions_idx)
        advantages = returns - values

        ppo_update(config.ppo_epochs, config.mini_batch_size, states, actions_idx, log_probs, returns, advantages)


if __name__ == "__main__":
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
                        'values':[1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]
                    },
                    'hidden1':{
                        'values':[64, 128, 256, 512]
                    },
                    'hidden2':{
                        'values':[64, 128, 256, 512]
                    }
                }
            }
        sweep_id = wandb.sweep(sweep_config, project='CutnFill_PPO')
        wandb.agent(sweep_id, train)

    else:
        train()

