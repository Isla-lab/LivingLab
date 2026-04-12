import torch
import torch.nn as nn

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List, Union

class Dynamics(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        """
        Reset `LivingLabEnv` dynamics to the initial state.
        """
        pass


class LSTMDynamics(Dynamics, nn.Module):
    def __init__(
            self,
            path: Union[Path, str],
            input_observation_names: List[str],
            input_norm_min: List[float],
            input_norm_max: List[float],
            num_layers: int,
            hidden_size: int,
            lookback: int,
            input_size: int=None,
            dropout: float=0.0
        ):
        Dynamics.__init__(self)
        nn.Module.__init__(self)
        
        # Input observations info
        assert len(input_observation_names) == len(input_norm_min) == len(input_norm_max), \
            f'`input_observation_names`, `input_norm_min` and `input_norm_max` must have the same length.' \
            + f'Found [{len(input_observation_names)}, {len(input_norm_min)}, {len(input_norm_max)}].'
        self.input_observation_names = input_observation_names
        self.input_norm_min = input_norm_min
        self.input_norm_max = input_norm_max

        # Model's structure values
        self.path = path
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lookback = lookback

        # Model's layer architecture
        self.l_lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers,batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.l_linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=1, # <- the predicted indoor temperature
        )
    
    @property
    def model_input(self):
        if not hasattr(self, '_model_input'):
            print('[WARN] LSTMDynamics input has not been initialized, yet. Call `.reset()`.')
            return None
        
        return self._model_input
    
    @property
    def hidden_state(self):
        if not hasattr(self, '_hidden_state'):
            print('[WARN] LSTMDynamics hidden state has not been initialized, yet. Call `.reset()`.')
            return None
        
        return self._hidden_state
    
    @property
    def input_size(self):
        return self._input_size
    
    @input_size.setter
    def input_size(self, new_size: int):
        assert new_size is None or new_size > 0, f'Invalid input dimensionality {new_size}. Must be either `None` or > 0.'
        self._input_size = len(self.input_observation_names) if new_size is None else new_size

    @model_input.setter
    def model_input(self, new_input):
        self._model_input = new_input

    @hidden_state.setter
    def hidden_state(self, new_h: Tuple[torch.Tensor, torch.Tensor]):
        self._hidden_state = new_h

    def forward(self, x: torch.Tensor, h: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        lstm_out, h = self.l_lstm(x, h)
        lstm_out = self.dropout(lstm_out)
        out = lstm_out[:, -1, :]
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h
    
    def reset(self):
        """
        Reset `LivingLabEnv` dynamics to the initial state.

        NOTE
        ----------
        Loads the model's state dict and resets both the hidden state and input tensor.
        """
        # Load state dict
        try:
            self.load_state_dict(torch.load(self.path)['model_state_dict'])        
        except RuntimeError:
            self.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['model_state_dict'])        
        except:
            self.load_state_dict(torch.load(self.path))

        # Reset hidden state and input
        self._hidden_state = self.init_hidden(1)
        self._model_input = [[None]*(self.lookback + 1) for _ in self.input_observation_names]

    def init_lstm(self) -> nn.LSTM:
        """
        Initalize Dynamics LSTM block.

        Returns
        ----------
        :return: the initialized LSTM block
        :rtype: torch.nn.LSTM
        """

        return nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
    
    def init_linear(self) -> nn.Linear:
        """
        Initalize Dynamics Linear layer for indoor temperature prediction.

        Returns
        ----------
        :return: the initialized Linear layer
        :rtype: torch.nn.Linear
        """

        return nn.Linear(
            in_features=self.hidden_size,
            out_features=1, # <- the predicted indoor temperature
        )
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize model's hidden state.

        Parameters
        ----------
        :param batch_size: Input batch size.
        :type batch_size: int

        Returns
        ----------
        :return: the initialized hidden state
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        return (hidden_state, cell_state)