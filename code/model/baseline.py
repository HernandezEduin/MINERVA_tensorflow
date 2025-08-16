from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
from typing import Union, Optional
from abc import ABC, abstractmethod

# tf.compat.v1.disable_eager_execution()


class baseline(ABC):
    """
    Abstract base class for baseline value estimation in reinforcement learning.
    
    Baselines are used to reduce variance in policy gradient methods by providing
    a reference value that represents the expected return from a given state.
    This helps stabilize training by centering the advantage estimates around zero.
    """
    
    @abstractmethod
    def get_baseline_value(self) -> Union[tf.Tensor, tf.Variable]:
        """
        Get the current baseline value estimate.
        
        Returns:
            Union[tf.Tensor, tf.Variable]: The baseline value used for variance reduction
                in policy gradient computation.
        """
        pass
    
    @abstractmethod
    def update(self, target: Union[tf.Tensor, float]) -> None:
        """
        Update the baseline estimate based on observed returns.
        
        Args:
            target (Union[tf.Tensor, float]): The target value (e.g., actual return)
                used to update the baseline estimate.
        """
        pass


class ReactiveBaseline(baseline):
    """
    A reactive baseline that maintains an exponentially weighted moving average.
    
    This baseline implementation uses a simple exponential moving average to track
    the expected return. It adapts quickly to changes in the reward distribution
    while maintaining stability through the learning rate parameter.
    
    The update rule is: b_new = (1 - α) * b_old + α * target
    where α is the learning rate that controls how quickly the baseline adapts.
    """
    
    def __init__(self, l: float) -> None:
        """
        Initialize the reactive baseline with a learning rate.
        
        Args:
            l (float): Learning rate (alpha) for the exponential moving average.
                Should be between 0 and 1, where higher values make the baseline
                more reactive to recent observations.
        """
        self.l: float = l
        self.b: tf.Variable = tf.Variable(0.0, trainable=False, name="baseline_value")
    
    def get_baseline_value(self) -> tf.Variable:
        """
        Get the current baseline value estimate.
        
        Returns:
            tf.Variable: The current baseline value maintained as an exponential
                moving average of observed returns.
        """
        return self.b
    
    def update(self, target: Union[tf.Tensor, float]) -> tf.Tensor:
        """
        Update the baseline using exponential moving average.
        
        Updates the baseline value using the formula:
        b_new = (1 - learning_rate) * b_old + learning_rate * target
        
        Args:
            target (Union[tf.Tensor, float]): The target return value used to
                update the baseline estimate.
                
        Returns:
            tf.Tensor: The updated baseline value after applying the moving average.
        """
        self.b = tf.add((1 - self.l) * self.b, self.l * target)
        return self.b
