# Manages training procedure
# Borrows heavily from the Udacity Deep Reinforcement Learning course

import numpy as np
import torch
from collections import deque


def dqn(env, agent, brain_name, n_episodes=500, max_t=310, eps_start=0.05, eps_end=0.01, eps_decay=0.995, update_target_every=2500):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode (this env stops after 303 anyway)
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        update_target_every (int): number of training timesteps between updates to target Q network
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_window_10 = deque(maxlen=10)  # last 10 scores
    eps = eps_start                    # initialize epsilon
    tt = 0
    max_score = -13
    for i_episode in range(1, n_episodes+1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            tt += 1
            if tt % update_target_every == 0:
                if not agent.ddqn:
                    print("\tUpdating target Q!")
                    agent.hard_update()
                else:
                    print("\t")
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores_window_10.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        max_score = max(score, max_score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tEps: {:.3f}\tScore: {:d}\tMax Score: {:d}\tAverage Score Last 10: {:.2f}\tAverage Score Last 100: {:.2f}'.format(i_episode, eps, int(score), int(max_score), np.mean(scores_window_10), np.mean(scores_window)), end="")
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tEps: {:.3f}\tAverage Score: {:.2f}'.format(i_episode-100, eps, np.mean(scores_window)))
            if not agent.ddqn:
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            else:
                torch.save(agent.qnetwork1.state_dict(), 'checkpoint_Q1.pth')
                torch.save(agent.qnetwork2.state_dict(), 'checkpoint_Q2.pth')
            break
    return scores