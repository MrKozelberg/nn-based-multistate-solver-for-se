# Imports
from imports import *
from configuration import configuration
from amplitude_functions import Gaussian
from neural_network import NeuralNetwork
from metropolis_algorithm import MetropolisAlgorithm

class TrialFuncion(nn.Module):
  """Trial function class"""

  def __init__(self, configuration: Configuration):
    """Initialize trial function"""
    super().__init__()
    self.defDevice(configuration)
    self.neuralNetwork = NeuralNetwork(configuration)
    self.defAmplitudeFunction(configuration)

  def defDevice(self, configuration: Configuration):
    """Set device"""    
    self.device = configuration.device

  def defAmplitudeFunction(self, configuration: Configuration):
    """Set amplitude function"""
    if configuration.amplitudeFunction == 'gaussian':
      self.amplitudeFunction = Gaussian()
    else:
      print("INVALID AMPLITUDE FUNCTION")
      sys.exit(1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    forward = self.neuralNetwork(x) * self.amplitudeFunction(x)
    return forward

  def gradient(self, x: torch.Tensor) -> torch.Tensor:
    gradient = (
      torch.zeros(
        (
          len(x),
          self.neuralNetwork.coordinateSpaceDim, 
          self.neuralNetwork.numberOfStates
        ), 
        device = self.device
      )
    )
    theAmplitudeFunction = self.amplitudeFunction(x)
    theAmplitudeFunctionGradient = self.amplitudeFunction.gradient(x)
    theNeuralNetwork = self.neuralNetwork(x)
    theNeuralNetworkGradient = self.neuralNetwork.gradient(x)
    for d in range(self.coordinateSpaceDim):
      gradient[:, d, :] = (
        theNeuralNetworkGradient[:, d, :] * theAmplitudeFunction
        + theNeuralNetwork * theAmplitudeFunctionGradient[:, d, :]
      )
    return gradient

  def laplacian(self, x: torch.Tensor) -> torch.Tensor:
    laplacian = (
      self.neuralNetwork.laplacian(x) * self.amplitudeFunction(x)
      + self.neuralNetwork(x) * self.amplitudeFunction.laplacian(x)
      + 2 * torch.sum(
        self.neuralNetwork.gradient(x) * self.amplitudeFunction.gradient(x),
        axis=1
      )
    )
    return laplacian

  def weightFunction(self, x: torch.Tensor) -> torch.Tensor:
    weightFunction = torch.mean(self.forward(x)**2, axis=1)
    return weightFunction

  def norm(self, x: torch.Tensor) -> torch.Tensor:
    norm = torch.zeros(
      self.neuralNetwork.numberOfStates,
      device=self.device
    )
    forward = self.forward(x)
    weightFunction = self.weightFunction(x)
    for stateNumber in range(self.neuralNetwork.numberOfStates):
      norm[stateNumber] = torch.mean(
        forward[:, stateNumber]**2 / weightFunction
      )
    return norm

  def spectrum(self, x: torch.Tensor) -> torch.Tensor:
    spectrum = torch.zeros(
      self.neuralNetwork.numberOfStates,
      device=self.device
    )
    forward = self.forward(x)
    laplacian = self.laplacian(x)
    weightFunction = self.weightFunction(x)
    potential = 0.5 * torch.sum(x*x, axis=1)
    for stateNumber in range(self.neuralNetwork.numberOfStates):
      spectrum[stateNumber] = torch.mean(
        forward[:, stateNumber] * (
          -0.5* laplacian[:, stateNumber] 
          + potential * forward[:, stateNumber]
        ) / weightFunction
      )
    spectrum = spectrum / self.norm(x)
    return spectrum
      





















