# Imports
from imports import *
from hyperparameters import Hyperparameters
from amplitude_functions import Gaussian
from neural_network import NeuralNetwork

class TrialFunction(nn.Module):
  """Trial function class"""

  def __init__(self, hyperparameters: Hyperparameters, name: str):
    """Initialize trial function"""
    super().__init__()
    self.name = name
    self.PATH = "../models/" + self.name + ".pt"
    self.defDevice(hyperparameters)
    self.neuralNetwork = NeuralNetwork(hyperparameters)
    self.defAmplitudeFunction(hyperparameters)
    # self.spectrum_values = torch.ones((hyperparameters.numberOfStates)) * 999.0

  def defDevice(self, hyperparameters: Hyperparameters):
    """Define device"""    
    self.device = hyperparameters.device

  def defAmplitudeFunction(self, hyperparameters: Hyperparameters):
    """Define amplitude function"""
    if hyperparameters.amplitudeFunction == 'gaussian':
      self.amplitudeFunction = Gaussian(hyperparameters)
    else:
      print("INVALID AMPLITUDE FUNCTION NAME")
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
    for coordinateNumber in range(self.neuralNetwork.coordinateSpaceDim):
      gradient[:, coordinateNumber, :] = (
        theNeuralNetworkGradient[:, coordinateNumber, :] * theAmplitudeFunction
        + theNeuralNetwork * theAmplitudeFunctionGradient[:, coordinateNumber, :]
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
    # weightFunction = torch.ones(
    #   (
    #     len(x),
    #     self.neuralNetwork.numberOfStates
    #   ),
    #   device=self.device
    # )
    # forward = self.forward(x).detach()
    # laplacian = self.laplacian(x).detach()
    # for stateNumber in range(self.neuralNetwork.numberOfStates):
    #   weightFunction[:,stateNumber] = (
    #       -0.5 * laplacian[:, stateNumber]
    #       + 0.5 * torch.sum(x*x, axis=1) * forward[:, stateNumber]
    #       - forward[:, stateNumber] * self.spectrum_values[stateNumber]
    #   )** 2
    # return weightFunction.mean(axis=1)
  
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
    # self.spectrum_values = spectrum.detach()
    return spectrum

  def totalSqueredResidual(self, x: torch.Tensor) -> torch.Tensor:
    forward = self.forward(x)
    laplacian = self.laplacian(x)
    weightFunction = self.weightFunction(x)
    spectrum = self.spectrum(x)
    norm = self.norm(x)
    return sum(
      [
        torch.mean(
          (
            -0.5 * laplacian[:, stateNumber]
            + 0.5 * forward[:, stateNumber] * torch.sum(x**2, axis=1)
            - forward[:, stateNumber] * spectrum[stateNumber]
          )** 2 / weightFunction
        ) / norm[stateNumber]
        for stateNumber in range(self.neuralNetwork.numberOfStates)
      ]
    )

  def squeredResidual(self, x: torch.Tensor, stateNumber: int) -> torch.Tensor:
    forward = self.forward(x)
    laplacian = self.laplacian(x)
    weightFunction = self.weightFunction(x)
    spectrum = self.spectrum(x)
    norm = self.norm(x)
    return torch.mean(
          (
            -0.5 * laplacian[:, stateNumber]
            + 0.5 * forward[:, stateNumber] * torch.sum(x**2, axis=1)
            - forward[:, stateNumber] * spectrum[stateNumber]
          )** 2 / weightFunction
        ) / norm[stateNumber]

  def totalNormalisationError(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sum((self.norm(x) - torch.tensor(1, device=self.device)) ** 2)

  def totalOrthogonalisationError(self, x: torch.Tensor) -> torch.Tensor:
    orthogonError = torch.zeros(
      (
        self.neuralNetwork.numberOfStates, 
        self.neuralNetwork.numberOfStates
      ),
      device=self.device
    )
    forward = self.forward(x)
    weightFunction = self.weightFunction(x)
    norm = self.norm(x)
    for stateNumber1 in range(1, self.neuralNetwork.numberOfStates):
      for stateNumber2 in range(0, stateNumber1):
        orthogonError[stateNumber1, stateNumber2] = (
          torch.square(
            torch.mean(
              forward[:, stateNumber1] 
              * forward[:, stateNumber2]
              / weightFunction
            )
          )
          / norm[stateNumber1]
          / norm[stateNumber2]
        )
    return torch.sum(orthogonError)





















