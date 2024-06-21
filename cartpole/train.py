from itertools import count
import argparse

import gymnasium as gym
import torch
from torch import nn
from torch.optim import AdamW

from memory import Transition, ReplayMemory
from model import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimize_model(args, policy_net, target_net, memory, optimizer):

    if len(memory) < args.batch:
        return

    # Get some sampling of transitions
    transitions = memory.sample(args.batch)

    # Transpose the batch (make it look good for the model
    batch = Transition(
        *zip(*transitions)
    )

    # Compute a mask of non-final states and concatenate the batch elements
    # Will look like tensor([True, True, False, ...]) where True if it is
    # non final and false if it is final
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=DEVICE, dtype=torch.bool
    )

    # Get next_states
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    # Get a batch of current states
    state_batch  = torch.cat(batch.state)
    # Get a batch of actions on that state
    action_batch = torch.cat(batch.action)
    # Get a batch of rewards from that action/state pair
    reward_batch = torch.cat(batch.reward)

    # policy_net returns a vector over all possible actions, using the gather function with action batch as input returns the
    # specific value for a specific action (if the action space has 2 options, action_batch selects 1 from those 2)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(args.batch, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()

    """What are we comparing here? 
    state_action_values is the actual value that the model got from making some choice
    expected_state_action_values is what the target_net thinks the next reward would be
    """
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def view(model):

    print("Viewing model")

    # Environment we train in
    test_env = gym.make("CartPole-v1", render_mode="human")

    # Get the number of state observations
    state, _ = test_env.reset()
    
    state = torch.tensor(state,
        dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # Work through episode
    for _ in count():

        # Select a random action from our model
        action = model.select_action(state, DEVICE)

        # Figure out the next state given that action
        observation, _, terminated, truncated, _ = test_env.step(action.item())

        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Move to the next state
        state = next_state

        if done:
            break

    test_env.close()

def main(args):

    # Environment we train in
    env = gym.make("CartPole-v1")

    # Get the number of state observations
    state, _ = env.reset()
    n_observations = len(state)
    
    # Instatiate policy and target networks
    policy_net = DQN(
        n_observations, env.action_space,
        args.eps_start, args.eps_end, args.eps_decay,
    ).to(DEVICE)
    target_net = DQN(
        n_observations, env.action_space,
        args.eps_start, args.eps_end, args.eps_decay,
    ).to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
    memory = ReplayMemory(10000)

    best_ep_reward = 0

    for i in range(args.episodes):

        print(f"E: {i+1}", end=" ")
        episode_reward = 0.

        # Reset env to init state, get this state as tensor
        state, _ = env.reset()
        state = torch.tensor(state,
            dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Work through episode
        for _ in count():

            # Select a random action from our model
            action = policy_net.select_action(state, DEVICE)

            # Figure out the next state given that action as well as reward and such
            observation, reward, terminated, truncated, _ = env.step(action.item())

            # I also want to insentivize the model into staying in the middle,
            # so there will be a punishment that is square with the distance to
            # the center of the platform
            cart_position = observation[0]
            cart_position = cart_position ** 2
            reward -= cart_position

            # Accumulate a reward and capture the reward for training
            episode_reward += float(reward)
            reward = torch.tensor([reward], device=DEVICE)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(args, policy_net, target_net, memory, optimizer)

            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*args.tau + target_net_state_dict[key]*(1-args.tau)
            
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break

        print(f"E reward: {episode_reward}")

        # Every 100 episodes, I want to view the model.
        if (i+1) % 100 == 0:
            view(policy_net)

            # If the reward exceeds the reward of any previous
            # models, save it out
            if episode_reward >= best_ep_reward:
                best_ep_reward = episode_reward
                torch.save(policy_net.state_dict(), "./best.pt")



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    """RL params"""
    parser.add_argument("--gamma",
        default=0.99, type=float,
        help="Discount factor for future reward compared to immediate reward"
    )
    parser.add_argument("--eps-start",
        default=0.9, type=float,
        help="Starting epsilon vlaue"
    )
    parser.add_argument("--eps-end",
        default=0.05, type=float,
        help="Final value of epsilon"
    )
    parser.add_argument("--eps-decay",
        default=1500, type=int,
        help="Controls the rate of exponential decay of epsilon (higher means slower)"
    )
    parser.add_argument("--tau",
        default=0.005, type=float,
        help="Update rate of target network"
    )
    parser.add_argument("--episodes",
        default=500, type=int,
        help="Number of episodes to work through"
    )

    """ML params"""
    parser.add_argument("--batch",
        default=128, type=int,
        help="Number of transitions sampled from the replay buffer"
    )
    parser.add_argument("--lr",
        default=1e-4, type=float,
        help="Learning rate of optimizer"
    )

    args = parser.parse_args()

    main(args)
