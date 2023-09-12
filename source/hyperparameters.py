# Imports
from imports import *

class Hyperparameters():
  """Configation class"""
  
  def __init__(self, hyperparametersFileName: str):
    """Initialize model hyperparameters"""
    self.defHyperparametersFromFile(hyperparametersFileName)
    self.defAndChooseDevice()

  def defAndChooseDevice(self):
    """Choose device"""
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
  
  def defHyperparametersFromFile(self, hyperparametersFileName: str):
    """Read model hyperparameters data from a data frame"""
    hyperparametersDataFrame = pd.read_csv(hyperparametersFileName)
    self.activationFunction = \
      hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='activation function' \
      ]['value'].to_string(index=False)
    self.amplitudeFunction = \
      hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='amplitude function' \
      ]['value'].to_string(index=False)
    self.hiddenLayerSize = \
      int(hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='hidden layer size' \
      ]['value'].to_string(index=False))
    self.coordinateSpaceDim = \
      int(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='coordinate space dimension' \
      ]['value'].to_string(index=False)
      )
    self.numberOfStates = \
      int(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='number of states' \
      ]['value'].to_string(index=False)
      )
    self.batchSize = \
      int(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='batch size' \
      ]['value'].to_string(index=False)
      )
    self.initialLearningRate = \
      float(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='initial learning rate' \
      ]['value'].to_string(index=False)
      )
    self.weightDecay = \
      float(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='weight decay' \
      ]['value'].to_string(index=False)
      )
    self.residualTermWeight = \
      float(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='residual term weight' \
      ]['value'].to_string(index=False)
      )
    self.normalisationTermWeight = \
      float(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='normalisation term weight' \
      ]['value'].to_string(index=False)
      )
    self.orthogonalisationTermWeight = \
      float(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='orthogonalisation term weight' \
      ]['value'].to_string(index=False)
      )
    self.energyTermWeight = \
      float(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='energy term weigth' \
      ]['value'].to_string(index=False)
      )
    self.metropolisAlgorithmParameter = \
      float(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='metropolis algorithm parameter' \
      ]['value'].to_string(index=False)
      )
    self.numberOfTrainingSteps = \
      int(
        hyperparametersDataFrame[ \
        hyperparametersDataFrame['name']=='number of training steps' \
      ]['value'].to_string(index=False)
      )
    