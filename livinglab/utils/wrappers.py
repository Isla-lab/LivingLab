import numpy as np
from gymnasium import spaces
from gymnasium import Wrapper, ObservationWrapper, ActionWrapper

from typing import List

from livinglab.envs.livinglab_env import LivingLabEnv


class NormalizedObservationWrapper(ObservationWrapper):
    """
    Wrapper for min-max and periodic normalization of environment observations.

    Parameters
    ----------
    :param env: A LivingLab environment.
    :type env: LivingLabEnv

    NOTE
    ----------
    1. Periodic temporal observations such as `hour`, `day_type` and `month`
       are periodically normalized usisng sin/cos transformations;
    2. All observations are min-max normalized within [0,1].
    """
    def __init__(self, env: LivingLabEnv):
        super().__init__(env=env)
        self.env: LivingLabEnv

        # Set the normalized observation space
        self.observation_space = self.env.unwrapped.estimate_observation_space(normalize=True)

    @property
    def observation_space(self)-> spaces.Box:
        """Return the space of normalized observations."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, new_space: spaces.Box):
        self._observation_space = new_space

    def observation(self, observations: np.ndarray) -> np.ndarray:
        """
        Return min-max normalized LivingLab observations.

        Parameters
        ----------
        :param observations: The observations returned from `LivingLabEnv.observations()`
        :type observations: np.ndarray

        Returns
        ----------
        :return: the min-max normalized observations.
        :rytype: np.ndarray

        NOTE
        ----------
        Input `observations` coming from `self.unwrapped` are currently not used for normalization.

        They are used to check the correct functioning of LivingLabEnv.
        """
        assert isinstance(observations, np.ndarray), f'Input `observations` should be a numpy array. Found {type(observations)}.'
        return self.unwrapped.observations(normalize=True)


class NormalizedActionWrapper(ActionWrapper):
    """
    Wrapper for min-max action normalization.

    Parameters
    ----------
    :param env: A LivingLab environment.
    :type env: LivingLabEnv

    NOTE
    ----------
    All actions are normalized within [0,1].
    """
    def __init__(self, env: LivingLabEnv):
        super().__init__(env=env)
        self.env: LivingLabEnv

        # Set the normalized action space
        self.action_space = spaces.Box(
            low=np.zeros(shape=self.unwrapped.action_space.low.size, dtype=np.float32),
            high=np.ones(shape=self.unwrapped.action_space.low.size, dtype=np.float32),
            dtype=np.float32
        )

    @property
    def action_space(self) -> spaces.Box:
        """Return the space of normalized actions."""
        return self._action_space
    
    @action_space.setter
    def action_space(self, new_space: spaces.Box):
        self._action_space = new_space

    def action(self, actions: np.ndarray) -> List[float]:
        """
        Return denormalized actions before calling `LivingLabEnv.step()`.

        Parameters
        ----------
        :param actions: The original normalized acitons.
        :type actions: np.ndarray

        Returns
        ----------
        :return: the list of denormalized actions to apply to the environment.
        :rtype: List[float]
        """
        transformed_actions = []
        for i, (l, h) in enumerate(zip(self.unwrapped.action_space.low, self.unwrapped.action_space.high)):
            a = actions[i]*(h - l) + l
            transformed_actions.append(a)

        return transformed_actions


class NormalizedSpaceWrapper(Wrapper):
    """
    Wrapper for normalized observation and action spaces.

    Parameters
    ----------
    :param env: A LivingLab environment.
    :type env: LivingLabEnv

    NOTE
    ----------
    Wraps `env` with `NormalizedObservationWrapper` and `NormalizedActionWrapper`.
    """
    def __init__(self, env: LivingLabEnv):
        # Wrap the environment and initialize super class
        env = NormalizedObservationWrapper(env)
        env = NormalizedActionWrapper(env)
        super().__init__(env)

        self.env: LivingLabEnv