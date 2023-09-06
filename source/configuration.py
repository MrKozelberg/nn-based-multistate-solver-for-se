# Imports
from imports import *

class Configuration():
  """Configation class"""
  
  def __init__(self, configurationFileName: str):
    """Initialize model configuration"""
    self.defConfigurationFromFile(configurationFileName)
    self.defAndChooseDevice()

  def defAndChooseDevice(self):
    """Choose device"""
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
  
  def defConfigurationFromFile(self, configurationFileName: str):
    """Read model configuration data from a data frame"""
    configurationDataFrame = pd.read_csv(configurationFileName)
    self.activationFunction = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='activation function' \
      ]['value'].to_string(index=False)
    self.amplitudeFunction = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='amplitude function' \
      ]['value'].to_string(index=False)
    self.hiddenLayerSize = \
      int(configurationDataFrame[ \
        configurationDataFrame['name']=='hidden layer size' \
      ]['value'].to_string(index=False))
    self.coordinateSpaceDim = \
      int(
        configurationDataFrame[ \
        configurationDataFrame['name']=='coordinate space dimension' \
      ]['value'].to_string(index=False)
      )
    self.numberOfStates = \
      int(
        configurationDataFrame[ \
        configurationDataFrame['name']=='number of states' \
      ]['value'].to_string(index=False)
      )
    self.batchSize = \
      int(
        configurationDataFrame[ \
        configurationDataFrame['name']=='batch size' \
      ]['value'].to_string(index=False)
      )
    self.initialLearningRate = \
      float(
        configurationDataFrame[ \
        configurationDataFrame['name']=='initial learning rate' \
      ]['value'].to_string(index=False)
      )
    self.weightDecay = \
      float(
        configurationDataFrame[ \
        configurationDataFrame['name']=='weight decay' \
      ]['value'].to_string(index=False)
      )
    self.residualTermWeight = \
      float(
        configurationDataFrame[ \
        configurationDataFrame['name']=='residual term weight' \
      ]['value'].to_string(index=False)
      )
    self.normalisationTermWeight = \
      float(
        configurationDataFrame[ \
        configurationDataFrame['name']=='normalisation term weight' \
      ]['value'].to_string(index=False)
      )
    self.orthogonalisationTermWeight = \
      float(
        configurationDataFrame[ \
        configurationDataFrame['name']=='orthogonalisation term weight' \
      ]['value'].to_string(index=False)
      )
    self.energyTermWeight = \
      float(
        configurationDataFrame[ \
        configurationDataFrame['name']=='energy term weigth' \
      ]['value'].to_string(index=False)
      )
    self.metropolisAlgorithmParameter = \
      float(
        configurationDataFrame[ \
        configurationDataFrame['name']=='metropolis algorithm parameter' \
      ]['value'].to_string(index=False)
      )
    