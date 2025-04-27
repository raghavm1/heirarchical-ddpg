import torch


class ObservationEncoder(torch.nn.Module):
    def initialize(
        self, observation_space, action_space=None,
        observation_normalizer=None, index=None
    ):
        self.observation_normalizer = observation_normalizer
        self.index = index
        print('index', index)
        if index is None:
            observation_size = observation_space.shape[0]
            self.index = 0
        else:
            observation_size = observation_space.shape[0] + index
        return observation_size

    def forward(self, observations):
        if self.observation_normalizer:
            print('observations shjape', observations.shape)
            # print('index', self.index)
            normalized_observations = self.observation_normalizer(observations[:,self.index:])
            observations[:,self.index:] = normalized_observations
        return observations


class ObservationActionEncoder(torch.nn.Module):
    def initialize(
        self, observation_space, action_space, observation_normalizer=None
    ):
        self.observation_normalizer = observation_normalizer
        observation_size = observation_space.shape[0]
        action_size = action_space.shape[0]
        return observation_size + action_size

    def forward(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return torch.cat([observations, actions], dim=-1)
