import gymnasium as gym
import torch
import numpy as np
from torchvision import transforms as T
from pathlib import Path
import datetime
from model import ZeroGameAgent
from network import ZeroGameNet
from logger import MetricLogger
import os
import sys
import csv
import matplotlib.pyplot as plt

def test_model(model, game_name="ALE/SpaceInvaders-v5", num_stacks=4):
    env = gym.make(game_name)
    env.reset()
    env = SkipFrame(env, skip=4, rewardmod=False)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = gym.wrappers.FrameStack(env, num_stack=num_stacks)
    total_reward = 0
    reps = 1
    model.net.eval()


    for i in range(reps):
        state = env.reset()
        done = False
        trunc = False
        while not done or trunc:
            
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=model.device).unsqueeze(0)
            action_values = model.net(state, model="online").cuda()
            action = torch.argmax(action_values, axis=1).item()

            try: 
                next_state, reward, done, trunc, info = env.step(action)
            
            except Exception as e:
                print(e)
                print("Error in env.step. Last action:", action)
                break

            state = next_state
            total_reward += reward

    avg_reward = total_reward / reps
    return avg_reward


class SkipFrame(gym.Wrapper):
    """
    Skips frames for every action. Returns the sum of the rewards for the skipped frames.
    
    :param env: (gym.Env) The environment
    :param skip: (int) The number of frames to skip
    """
    def __init__(self, env, skip, rewardmod=True):
        super().__init__(env)
        self._skip = skip
        self.rewardmod = rewardmod
    
    def step(self, action):
        """
        Repeat action, and sum reward
        
        :param action: (int) The action
        :return: (tuple) The new observation, the sum of the rewards, the done flag, and additional information
        """
        total_reward = 0.0
        lives = self.env.unwrapped.ale.lives()
        
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            if self.rewardmod:
                new_lives = info.get("lives", 0)

                if new_lives < lives:
                    reward -= 60
                
                if done:
                    reward -= 80
                
                lives = new_lives

            total_reward += reward
            

            if done:
                break

        return obs, total_reward, done, trunk, info
    

class GrayScaleObservation(gym.ObservationWrapper):
    """
    Converts the observation to grayscale
    
    :param env: (gym.Env) The environment
    """
    
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        """
        Turns Height, Width, Color array into Color, Height, Width tensor

        :param observation: (np.ndarray) The observation
        """
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
    
if __name__ == "__main__":

    env = gym.make("ALE/SpaceInvaders-v5")
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))

    num_stacks = 4

    env = gym.wrappers.FrameStack(env, num_stack=num_stacks)
    actions = env.action_space.n
    
    dirs = ["['SpaceInvaders']", "['DemonAttack', 'SpaceInvaders']", "['Carnival', 'SpaceInvaders']", "['AirRaid', 'SpaceInvaders']"]

    for dir in dirs:
        path = Path(f"checkpoints/{dir}")
        game = os.listdir(path)[-1]
        for model in os.listdir(path / game):
            path_to_model = (os.path.dirname(__file__)/path/game/model)
            print(path_to_model)
            
            fifty_game_avg = {}

            i = 1_000_000
            checkpoint = path_to_model / f"{i}.chkpt"
            zg = ZeroGameAgent(state_space=(num_stacks, 84, 84), action_space=actions)
            zg.net.load_state_dict(torch.load(checkpoint)["model"])
            
            for j in range(1000): 
                avg_reward = test_model(zg)
                fifty_game_avg[j+1] = avg_reward
                
            #save fifty_game_avg to csv
            with open(path_to_model / "500games.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(fifty_game_avg.items())