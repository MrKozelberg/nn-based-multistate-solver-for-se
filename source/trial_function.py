# Imports
from imports import *
from configuration import configuration
from amplitude_functions import Gaussian
from neural_network import NeuralNetwork

class TrialFuncion(nn.Module):
  """Trial function class"""

  def __init__(self, configuration: Configuration):
    """Initialize trial function"""
    super().__init__()
    self.neuralNetwork = NeuralNetwork(configuration)
    self.setAmplitudeFunction(configuration)
    self.setDevice(configuration)

  def setDevice(self, configuration: Configuration):
    """Set device"""    
    self.device = configuration.device

  def setAmplitudeFunction(self, configuration: Configuration):
    """Set amplitude function"""
    if configuration.amplitudeFunction == 'gaussian':
      self.amplitudeFunction = Gaussian()
    else:
      print("INVALID AMPLITUDE FUNCTION")
      sys.exit(1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    
