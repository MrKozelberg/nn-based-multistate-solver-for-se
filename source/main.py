# Imports
from trial_function import *


def testTrialFunction():
    """Checks if TrialFunction class works or not"""
    fillParameters(
        AF="sin",  # Activation function
        Nh=int(2**5),  # Number of nodes in each hidden layer
        D=2,  # Dimension of the coordinate space
        M=16,  # Number of states we want to find
        std=7,  # Standard deviation of sample distribution
        B=int(2**11),  # Batch size
        learningRate=1e-3,  # Starting value of the learning rate (we use ADAM
        # optimizer)
        weightDecay=1e-3,  # Weight decay (we use ADAM optimizer)
        wRMax=1e3,  # Constraint on a value of a reversal residual term from a
        # loss function
        wAMax=1e3,  # Constraint on a value of a reversal normalisation term
        # from a loss function
        wBMax=1e3,  # Constraint on a value of a reversal orthogonalisation
        # term from a loss function
        wEMax=1e3,  # Constraint on a value of a reversal energy term from
        # a loss function
        wEpoch=200,  # How many steps will be done with the same loss function
        # weights
    )
    tf = TrialFunction()
    x = torch.ones((3, parameters["D"])).to(device)
    print(tf(x))
    print(tf.gradient(x))
    print(tf.laplacian(x))
    print(tf.spectrum(x))


if __name__ == "__main__":
    # testTrialFunction()
    fillParameters(
        AF="sin",  # Activation function
        Nh=int(2**5),  # Number of nodes in each hidden layer
        D=2,  # Dimension of the coordinate space
        M=9,  # Number of states we want to find
        std=7,  # Standard deviation of sample distribution
        B=int(2**12),  # Batch size
        learningRate=1e-3,  # Starting value of the learning rate (we use ADAM
        # optimizer)
        weightDecay=1e-3,  # Weight decay (we use ADAM optimizer)
        wRMax=1e3,  # Constraint on a value of a reversal residual term from a
        # loss function
        wAMax=1e3,  # Constraint on a value of a reversal normalisation term
        # from a loss function
        wBMax=1e9,  # Constraint on a value of a reversal orthogonalisation
        # term from a loss function
        wEMax=1e3,  # Constraint on a value of a reversal energy term from
        # a loss function
        wEpoch=100,  # How many steps will be done with the same loss function
        # weights
    )
    tf = TrialFunction()
    history = training(
        trialFunction=tf,  # Trial function we will train
        steps=1000,  # Number of steps
    )
