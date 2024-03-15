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



class SkipFrame(gym.Wrapper):
    """
    Skips frames for every action. Returns the sum of the rewards for the skipped frames.
    
    :param env: (gym.Env) The environment
    :param skip: (int) The number of frames to skip
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    
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
            new_lives = info.get("lives", 0)

            if new_lives < lives:
                reward -= 40

            total_reward += reward
            lives = new_lives

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
    
def test_model(model, num_stacks=4):
    env = gym.make("ALE/SpaceInvaders-v5")
    env.reset()
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = gym.wrappers.FrameStack(env, num_stack=num_stacks)
    total_reward = 0

    for i in range(50):
        state = env.reset()
        done = False
        while not done:
            action = model.act_network_only(state)
            next_state, reward, done, trunc, info = env.step(action)
            state = next_state
            total_reward += reward

    avg_reward = total_reward / 50
    return avg_reward
        


def train(path = None, epsilon = None):
    env = gym.make("ALE/SpaceInvaders-v5")
    actions = env.action_space.n

    env.reset()

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(84, 84))

    num_stacks = 4

    env = gym.wrappers.FrameStack(env, num_stack=num_stacks)

    episodes = 8000000
    
    save_dir = Path(os.path.dirname(__file__)) / Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    zg = ZeroGameAgent(state_space=(num_stacks, 84, 84), action_space=actions, save_dir=save_dir)
    
    if path:
        zg.net.load_state_dict(torch.load(path)["model"])

    if epsilon:
        zg.exploration_rate = epsilon

        
    logger = MetricLogger(save_dir)

    for e in range(episodes):
    
        state = env.reset()
        lives = 3

        # Play the game!
        while True:

            # Run agent on the state
            action = zg.act(state)

            # Agent performs action
            try:
                next_state, reward, done, trunc, info = env.step(action)
                

            except Exception as e:
                print(e)
                print("Error in env.step. Last action:", action)
                break

            # Remember
            zg.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = zg.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done:
                break
            
        logger.log_episode()
        
        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=zg.exploration_rate, step=zg.curr_step)
        
        if (e % 500==0) and (e >= 10000):
            avg_reward = test_model(zg, num_stacks)
            print(f"Average reward over 50 episodes: {avg_reward}")
            if avg_reward >= 1652:
                print("Model has learned to play the game!")
                zg.save(save_name="final")
                break

if __name__ == "__main__":
    print(torch.cuda.is_available())
    path = Path(__file__).parent / Path("checkpoints") / "2024-03-13T20-04-45" / "test_net_13.chkpt"
    epsilon =  0.3  
    train(path = path, epsilon = epsilon)