from itertools import count
import argparse

import gymnasium as gym
import torch

from model import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    # Environment we train in
    env = gym.make("LunarLander-v2",
        render_mode="human"
    )

    # Get the number of state observations
    state, _ = env.reset()
    n_observations = len(state)
    
    # Instatiate policy and target networks
    policy_net = DQN(
        n_observations, env.action_space,
        args.eps_start, args.eps_end, args.eps_decay,
    ).to(DEVICE)

    policy_net.load_state_dict(torch.load(args.model))

    state = torch.tensor(state,
        dtype=torch.float32, device=DEVICE).unsqueeze(0)

    total_reward = 0.

    # Work through episode
    for _ in count():

        # Select a random action from our model
        action = policy_net.select_action(state, DEVICE)

        # Figure out the next state given that action
        observation, reward, terminated, truncated, _ = env.step(action.item())

        total_reward += float(reward)

        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Move to the next state
        state = next_state

        if done:
            break

    print(f"Total reward: {total_reward}")

    env.close()


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    """
    RL params
    EPS is set to 0 so that the model never chooses random action
    """
    parser.add_argument("--eps-start",
        default=0., type=float,
        help="Starting epsilon vlaue"
    )
    parser.add_argument("--eps-end",
        default=0., type=float,
        help="Final value of epsilon"
    )
    parser.add_argument("--eps-decay",
        default=1, type=int,
        help="Controls the rate of exponential decay of epsilon (higher means slower)"
    )

    """File path to model"""
    parser.add_argument("--model",
        required=True, type=str,
        help="File path to a trained model"
    )

    args = parser.parse_args()

    main(args)
