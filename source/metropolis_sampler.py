# Imports
from imports import *
from hyperparameters import Hyperparameters

class MetropolisSampler():
  """Class for Metropolis sampler"""

  def __init__(self, hyperparameters: Hyperparameters):
    self.epsilon = hyperparameters.metropolisAlgorithmParameter
    self.device = hyperparameters.device
    self.sample = self.initialSample(hyperparameters)

  def initialSample(self, hyperparameters: Hyperparameters) -> torch.Tensor:
    return 3 * (
      torch.rand(
        (
          hyperparameters.batchSize, 
          hyperparameters.coordinateSpaceDim
        )
      ) - 0.5
    ).to(self.device)
  
  def updateSampleBasOnDistrDens(self, distributionDensity):
    newSample = self.sample + self.epsilon * (
      2 * torch.rand_like(
        self.sample, 
        device=self.device
      ) - 1
    )
    critVal = distributionDensity(newSample) / distributionDensity(self.sample)
    doesPointMove = (torch.rand(len(self.sample), device=self.device) <= critVal)
    newSample = (
      torch.mul(doesPointMove.int(), newSample.t()).t()
      + torch.mul((1 - doesPointMove.int()), self.sample.t()).t()
    )
    self.sample = newSample