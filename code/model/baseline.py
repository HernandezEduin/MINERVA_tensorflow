"""
Baseline estimation for variance reduction in policy gradient reinforcement learning.

This module provides baseline estimators that reduce variance in policy gradient
methods by subtracting expected value estimates from returns. Baselines are crucial
for stable training in REINFORCE and related algorithms, as they center advantage
estimates around zero without introducing bias.

The baseline serves as a learned estimate of the state value function V(s),
which when subtracted from returns R_t creates advantage estimates A_t = R_t - V(s).
This variance reduction technique significantly improves sample efficiency and
training stability in policy gradient methods.

Key components:
- Abstract baseline interface for different estimation strategies
- ReactiveBaseline: Exponential moving average implementation
- Integration with TensorFlow for gradient-based optimization
- Support for both tensor and scalar target updates

Classes:
    baseline: Abstract base class defining the baseline interface
    ReactiveBaseline: Exponential moving average baseline implementation
"""

from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from typing import Union, Optional
from abc import ABC, abstractmethod

class baseline(ABC):
    """
    Abstract base class for baseline value estimation in reinforcement learning.
    
    Defines the interface for baseline estimators used in policy gradient methods
    to reduce variance by providing state value estimates. Baselines estimate the
    expected return V(s) from each state, which when subtracted from actual returns
    creates advantage estimates that center around zero.
    
    The baseline is critical for:
    - Variance reduction in policy gradient estimates
    - Improved sample efficiency during training
    - More stable convergence in REINFORCE algorithms
    - Unbiased advantage estimation (baselines don't affect expected gradients)
    
    Implementations should provide efficient update mechanisms and integration
    with TensorFlow's computational graph for gradient-based optimization.
    
    Example:
        >>> baseline = ReactiveBaseline(l=0.1)
        >>> current_value = baseline.get_baseline_value()
        >>> updated_value = baseline.update(observed_return)
    """
    
    @abstractmethod
    def get_baseline_value(self) -> Union[tf.Tensor, tf.Variable]:
        """
        Retrieve the current baseline value estimate.
        
        Returns the baseline's current estimate of expected returns, which is used
        for variance reduction in policy gradient computations. The baseline value
        represents an approximation of the state value function V(s).
        
        Returns:
            Current baseline value estimate as a TensorFlow tensor or variable.
            This value is subtracted from returns to compute advantages in
            policy gradient algorithms.
            
        Note:
            - Should return a scalar value or tensor compatible with return shapes
            - Must be differentiable if used in gradient-based baseline updates
            - Called during both training and inference phases
        """
        pass
    
    @abstractmethod
    def update(self, target: Union[tf.Tensor, float]) -> Union[tf.Tensor, None]:
        """
        Update the baseline estimate using observed target values.
        
        Incorporates new observations (typically actual returns or value estimates)
        to improve the baseline's approximation of expected returns. The update
        mechanism depends on the specific baseline implementation but should
        reduce prediction error over time.
        
        Args:
            target: The observed target value used for baseline updates.
                   Typically actual returns R_t, temporal difference targets,
                   or other value estimates. Can be scalar or tensor.
                   
        Returns:
            Updated baseline value or None if update is performed in-place.
            Some implementations may return the new baseline value for
            monitoring or further computation.
            
        Note:
            - Should be called after each episode or batch of experiences
            - Target values should represent unbiased estimates of expected returns
            - Update frequency affects baseline adaptation speed vs stability
        """
        pass


class ReactiveBaseline(baseline):
    """
    Reactive baseline using exponentially weighted moving average for value estimation.
    
    Implements a simple yet effective baseline that maintains an exponential moving
    average of observed returns. This approach provides a good balance between
    adaptation speed and stability, making it suitable for environments with
    slowly changing reward distributions.
    
    The baseline adapts according to the update rule:
    b_new = (1 - α) * b_old + α * target
    
    Where:
    - α (learning rate) controls adaptation speed
    - Higher α values make the baseline more reactive to recent observations
    - Lower α values provide more stable estimates with slower adaptation
    
    Advantages:
    - Simple and computationally efficient
    - Automatic adaptation to changing reward scales
    - Requires minimal hyperparameter tuning
    - Integrates seamlessly with TensorFlow computational graphs
    
    Attributes:
        l (float): Learning rate parameter controlling adaptation speed
        b (tf.Variable): Current baseline value maintained as TensorFlow variable
        
    Example:
        >>> baseline = ReactiveBaseline(l=0.1)  # Slow adaptation
        >>> baseline_fast = ReactiveBaseline(l=0.5)  # Fast adaptation
        >>> current_estimate = baseline.get_baseline_value()
        >>> baseline.update(observed_return)
    """
    
    def __init__(self, l: float) -> None:
        """
        Initialize the reactive baseline with specified learning rate.
        
        Creates a new baseline estimator with an exponential moving average
        update rule. The baseline value is initialized to zero and stored
        as a non-trainable TensorFlow variable for efficient computation.
        
        Args:
            l: Learning rate (alpha) controlling the exponential moving average.
               Should be in range (0, 1] where:
               - Values near 0: Slow adaptation, stable estimates
               - Values near 1: Fast adaptation, more reactive to recent data
               - Typical values: 0.01 - 0.1 for stable training
               
        Note:
            - Baseline variable is marked as non-trainable since updates
              are performed explicitly through the update() method
            - Initial baseline value of 0.0 works well for most applications
            - Learning rate can be adjusted during training if needed
        """
        self.l: float = l
        self.b: tf.Variable = tf.Variable(0.0, trainable=False, name="baseline_value")
    
    def get_baseline_value(self) -> tf.Variable:
        """
        Retrieve the current baseline value for variance reduction.
        
        Returns the current exponential moving average estimate maintained
        by this baseline. This value represents the baseline's best estimate
        of expected returns and is used for computing advantages in policy
        gradient algorithms.
        
        Returns:
            Current baseline value as a TensorFlow variable. This scalar value
            can be directly used in policy gradient computations for variance
            reduction without introducing bias to the gradient estimates.
            
        Note:
            - Returns the same tf.Variable instance for computational efficiency
            - Value updates automatically via the update() method
            - Can be used multiple times within the same computational graph
        """
        return self.b
    
    def update(self, target: Union[tf.Tensor, float]) -> tf.Tensor:
        """
        Update baseline using exponential moving average with observed target.
        
        Performs an exponential moving average update to incorporate new
        observations into the baseline estimate. This update rule provides
        a good balance between stability and adaptation to changing environments.
        
        The update follows the formula:
        b_new = (1 - learning_rate) * b_old + learning_rate * target
        
        Args:
            target: The target value used for updating the baseline estimate.
                   Typically the observed return or value estimate from the
                   current episode. Can be a scalar float or TensorFlow tensor.
                   
        Returns:
            The updated baseline value after applying the exponential moving
            average update. This is the same as calling get_baseline_value()
            after the update completes.
            
        Note:
            - Update is performed in-place on the baseline variable
            - Learning rate controls how much the baseline changes per update
            - Should be called once per episode or batch for proper averaging
            - Target values should be unbiased estimates of expected returns
        """
        self.b = tf.add((1 - self.l) * self.b, self.l * target)
        return self.b
