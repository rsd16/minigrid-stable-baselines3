
# you can also check: https://github.com/BolunDai0216/MinigridMiniworldTransfer/tree/main
# it has useful tools and stuff.


import minigrid
import torch

import gymnasium as gym
import torch.nn as nn

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
#from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.env_util import make_vec_env


env = gym.make("MiniGrid-Empty-5x5-v0")

env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

#env = FlatObsWrapper(env)

#check_env(env, warn=True)

env = Monitor(env, "./a2c_cartpole_tensorboard/")

state, _ = env.reset()
#print(state)

#######################################################################################

# Simple API, default networks...

model = PPO("CnnPolicy", env, tensorboard_log="./a2c_cartpole_tensorboard/", verbose=1)
#model.learn(total_timesteps=10, tb_log_name="first_run")
##model.learn(total_timesteps=10, tb_log_name="second_run", reset_num_timesteps=False)

#print(model.policy)

env = Monitor(env)
mean_reward, std_reward = evaluate_policy(model, env)
print(f"{mean_reward = }, {std_reward = }")

#######################################################################################

# Custom Feature Extractor ... still using CNNPolicy.
# but i don't remember if i changed the feature extractor, then wtf is the cnnpolicy?
# because last layer is features_dim, which is 256.


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))


policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=128))
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="./a2c_cartpole_tensorboard/", verbose=1)
#model.learn(total_timesteps=10000, tb_log_name="first_run")

env = Monitor(env)
mean_reward, std_reward = evaluate_policy(model, env)
print(f"{mean_reward = }, {std_reward = }")

#######################################################################################

# This is actual custom network.


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self, feature_dim, last_layer_dim_pi=64, last_layer_dim_vf=64):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.policy_net(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.policy_linear = nn.Sequential(nn.Linear(n_flatten, last_layer_dim_pi), nn.ReLU())

        self.value_net = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.value_net(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.value_linear = nn.Sequential(nn.Linear(n_flatten, last_layer_dim_vf), nn.ReLU())

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_linear(self.policy_net(features))

    def forward_critic(self, features):
        return self.value_linear(self.value_net(features))


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        kwargs["ortho_init"] = False

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_cnn_extractor(self):
        self.cnn_extractor = CustomNetwork(self.features_dim)


model = PPO(CustomActorCriticPolicy, env, tensorboard_log="./a2c_cartpole_tensorboard/", verbose=1)
#model.learn(total_timesteps=10000, tb_log_name="first_run")

env = Monitor(env)
mean_reward, std_reward = evaluate_policy(model, env)
print(f"{mean_reward = }, {std_reward = }")

#######################################################################################


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "direction":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16
            elif key == "mission":
                pass

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


env = gym.make("MiniGrid-Empty-5x5-v0")
state, _ = env.reset()
print(state)
sdf
policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor, features_extractor_kwargs=dict(features_dim=128))
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="./a2c_cartpole_tensorboard/", verbose=1)
#model.learn(total_timesteps=10000, tb_log_name="first_run")

env = Monitor(env)
mean_reward, std_reward = evaluate_policy(model, env)
print(f"{mean_reward = }, {std_reward = }")