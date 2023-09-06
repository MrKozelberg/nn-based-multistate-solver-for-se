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
      configurationDataFrame[ \
        configurationDataFrame['name']=='hidden layer size' \
      ]['value'].to_string(index=False)
    self.coordinateSpaceDim = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='coordinate space dimension' \
      ]['value'].to_string(index=False)
    self.numberOfStates = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='number of states' \
      ]['value'].to_string(index=False)
    self.batchSize = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='batch size' \
      ]['value'].to_string(index=False)
    self.initialLearningRate = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='initial learning rate' \
      ]['value'].to_string(index=False)
    self.weightDecay = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='weight decay' \
      ]['value'].to_string(index=False)
    self.residualTermWeight = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='residual term weight' \
      ]['value'].to_string(index=False)
    self.normalisationTermWeight = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='normalisation term weight' \
      ]['value'].to_string(index=False)
    self.orthogonalisationTermWeight = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='orthogonalisation term weight' \
      ]['value'].to_string(index=False)
    self.energyTermWeight = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='energy term weigth' \
      ]['value'].to_string(index=False)
    self.metropolisAlgorithmParameter = \
      configurationDataFrame[ \
        configurationDataFrame['name']=='metropolis algorithm parameter' \
      ]['value'].to_string(index=False)
    