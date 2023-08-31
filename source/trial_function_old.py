#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:11:56 2023

@author: akozlov
"""

# Imports
# System
import sys

# Time
import time

# Date time
import datetime as dt

# PyTorch framework
import torch
from torch import nn

# NumPy
import numpy as np

# Pandas
import pandas as pd

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {device}: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("No GPU found, using cpu")


# def chooseDevice():
#     """Chooses device"""
#     global device
#     # Checks to see if gpu is available. If it is, use it else use cpu


# Dictionary of model parameters
parameters = {}


def fillParameters(
    AF: str,  # Activation function
    AMPLITUDE: str,  # Amplitude function
    Nh: int,  # Number of nodes in each hidden layer
    D: int,  # Dimension of the coordinate space
    M: int,  # Number of states we want to find
    std: float,  # Standard deviation of sample distribution
    B: int,  # Batch size
    learningRate: float,  # Starting value of the learning rate (we use ADAM
    # optimizer)
    weightDecay: float,  # Weight decay (we use ADAM optimizer)
    # wR: float,  # Residual term weight
    # wA: float,  # Normalisation term weight
    # wB: float,  # Orthogonalisation term weight
    # wE: float,  # Energy term weight
    wRMax: float,  # Constraint on a value of a reversal residual term from a
    # loss function
    wAMax: float,  # Constraint on a value of a reversal normalisation term
    # from a loss function
    wBMax: float,  # Constraint on a value of a reversal orthogonalisation
    # term from a loss function
    wEMax: float,  # Constraint on a value of a reversal energy term from
    # a loss function
    wEpoch: int,  # How many steps will be done with the same loss function
    # weights
):
    """Fills dictionary of model parameters"""
    global parameters
    parameters["AF"] = AF
    parameters["AMPLITUDE"] = AMPLITUDE
    parameters["Nh"] = Nh
    parameters["D"] = D
    parameters["M"] = M
    parameters["std"] = std
    parameters["B"] = B
    parameters["learningRate"] = learningRate
    parameters["weightDecay"] = weightDecay
    # parameters["wR"] = wR
    # parameters["wA"] = wA
    # parameters["wB"] = wB
    # parameters["wE"] = wE
    parameters["wRMax"] = wRMax
    parameters["wAMax"] = wAMax
    parameters["wBMax"] = wBMax
    parameters["wEMax"] = wEMax
    parameters["wEpoch"] = wEpoch
    s = ""
    for key in parameters.keys():
        if key != "configuration":
            s = s + f"{key}{parameters[key]}"
    parameters["configuration"] = s  # Model configuration


class Sin(nn.Module):
    """Custom sin activation function class"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

    def diff(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)

    def diff2(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.sin(x)


class Tanh(nn.Module):
    """Custom tanh activation function class"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def diff(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / torch.cosh(x) ** 2

    def diff2(self, x: torch.Tensor) -> torch.Tensor:
        return -2 * torch.sinh(x) / torch.cosh(x) ** 3


class Gaussian(nn.Module):
    """Custom Gaussian amplitude function class"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.ones((x.shape[0], parameters["M"])).to(device)
        result = torch.mul(
            result.t(), torch.exp(-torch.sum(x**2, axis=1) / 2)
        ).t()
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["M"]
        ), "Wrong shape 1 of the result"
        return result

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.ones((x.shape[0], parameters["D"], parameters["M"])).to(
            device
        )
        for d in range(parameters["D"]):
            # first term
            a = torch.ones((x.shape[0], parameters["M"])).to(device)
            a = torch.mul(
                a.t(),
                torch.exp(-torch.sum(x**2, axis=1) / 2) * (-2 * x[:, d] / 2),
            ).t()
            # keeping result
            result[:, d, :] = a
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["D"]
        ), "Wrong shape 1 of the result"
        assert (
            result.shape[2] == parameters["M"]
        ), "Wrong shape 2 of the result"
        return result

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.ones((x.shape[0], parameters["M"])).to(device)
        result = torch.mul(
            result.t(),
            torch.exp(-torch.sum(x**2, axis=1) / 2)
            * (
                torch.sum((-2 * x / 2) ** 2, axis=1)
                - 2 * torch.sum(torch.ones_like(x) / 2, axis=1)
            ),
        ).t()
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["M"]
        ), "Wrong shape 1 of the result"
        return result


class Exponent4(nn.Module):
    """Custom Exponent amplitude function class"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.ones((x.shape[0], parameters["M"])).to(device)
        result = torch.mul(
            result.t(),
            torch.exp(-torch.sum(x**4, axis=1) / parameters["std"] ** 4),
        ).t()
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["M"]
        ), "Wrong shape 1 of the result"
        return result

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.ones((x.shape[0], parameters["D"], parameters["M"])).to(
            device
        )
        for d in range(parameters["D"]):
            # first term
            a = torch.ones((x.shape[0], parameters["M"])).to(device)
            a = torch.mul(
                a.t(),
                torch.exp(-torch.sum(x**4, axis=1) / parameters["std"] ** 4)
                * (-4 * x[:, d] ** 3 / parameters["std"] ** 4),
            ).t()
            # keeping result
            result[:, d, :] = a
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["D"]
        ), "Wrong shape 1 of the result"
        assert (
            result.shape[2] == parameters["M"]
        ), "Wrong shape 2 of the result"
        return result

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.ones((x.shape[0], parameters["M"])).to(device)
        result = torch.mul(
            result.t(),
            torch.exp(-torch.sum(x**4, axis=1) / parameters["std"] ** 4)
            * (
                torch.sum((-4 * x**3 / parameters["std"] ** 4) ** 2, axis=1)
                - 12 * torch.sum(x**2 / parameters["std"] ** 4, axis=1)
            ),
        ).t()
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["M"]
        ), "Wrong shape 1 of the result"
        return result


class NN(nn.Module):
    """Class for a neural network"""

    def __init__(self):
        super().__init__()
        global parameters
        if parameters["AF"] == "sin":
            self.AF = Sin()
        elif parameters["AF"] == "tanh":
            self.AF = Tanh()
        else:
            print("Wrong AF")
            sys.exit(1)
        # NUMBER OF HIDDEN LAYERS IS ASSUMED TO BE 3
        self.stack = nn.Sequential(
            nn.Linear(parameters["D"], parameters["Nh"]),
            self.AF,
            nn.Linear(parameters["Nh"], parameters["Nh"]),
            self.AF,
            nn.Linear(parameters["Nh"], parameters["Nh"]),
            self.AF,
            nn.Linear(parameters["Nh"], parameters["M"]),
            self.AF,
        )
        for i in range(len(self.stack)):
            if "weight" in dir(self.stack[i]):
                torch.nn.init.normal_(self.stack[i].weight, 0, np.sqrt(0.1))

        global device
        # chooseDevice()
        self.stack.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = self.stack(x).to(device)
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["M"]
        ), "Wrong shape 1 of the result"
        return result

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.ones((x.shape[0], parameters["D"], parameters["M"])).to(
            device
        )
        for d in range(parameters["D"]):
            ydiff = self.AF.diff(self.stack[0](x)) * self.stack[0].weight[:, d]
            y = self.stack[0 * 2 + 1](self.stack[0 * 2](x))
            # NUMBER OF HIDDEN LAYERS IS ASSUMED TO BE 3
            for i in range(1, 3 + 1):
                ydiff = self.AF.diff(self.stack[i * 2](y)) * torch.matmul(
                    ydiff, self.stack[i * 2].weight.t()
                )
                y = self.stack[i * 2 + 1](self.stack[i * 2](y))
            # keeping result
            result[:, d, :] = ydiff
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["D"]
        ), "Wrong shape 1 of the result"
        assert (
            result.shape[2] == parameters["M"]
        ), "Wrong shape 2 of the result"
        return result

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        preresult = torch.ones(
            (x.shape[0], parameters["D"], parameters["M"])
        ).to(device)
        for d in range(parameters["D"]):
            ydiff = self.AF.diff(self.stack[0](x)) * self.stack[0].weight[:, d]
            ydiff2 = (
                self.AF.diff2(self.stack[0](x))
                * self.stack[0].weight[:, d] ** 2
            )
            y = self.stack[0 * 2 + 1](self.stack[0 * 2](x))
            # NUMBER OF HIDDEN LAYERS IS ASSUMED TO BE 3
            for i in range(1, 3 + 1):
                # second derivative
                ydiff2 = self.AF.diff2(self.stack[i * 2](y)) * torch.matmul(
                    ydiff, self.stack[i * 2].weight.t()
                ) ** 2 + self.AF.diff(self.stack[i * 2](y)) * torch.matmul(
                    ydiff2, self.stack[i * 2].weight.t()
                )
                # first derivative
                ydiff = self.AF.diff(self.stack[i * 2](y)) * torch.matmul(
                    ydiff, self.stack[i * 2].weight.t()
                )
                # output of the current layer
                y = self.stack[i * 2 + 1](self.stack[i * 2](y))
            # Keeping preresult
            preresult[:, d, :] = ydiff2
        result = torch.sum(preresult, axis=1)
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["M"]
        ), "Wrong shape 1 of the result"
        return result


class TrialFunction(nn.Module):
    """Trial function class"""

    def __init__(self):
        super().__init__()
        self.nn = NN()
        global parameters
        if parameters["AMPLITUDE"] == "gaussian":
            self.amplitude = Gaussian()
        elif parameters["AMPLITUDE"] == "exponent4":
            self.amplitude = Exponent4()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = self.nn(x).to(device) * self.amplitude(x).to(device)
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["M"]
        ), "Wrong shape 1 of the result"
        return result

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.ones((x.shape[0], parameters["D"], parameters["M"])).to(
            device
        )
        a = self.amplitude(x)
        ag = self.amplitude.gradient(x)
        nn = self.nn(x)
        nng = self.nn.gradient(x)
        for d in range(parameters["D"]):
            result[:, d, :] = nng[:, d, :] * a + nn * ag[:, d, :]
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["D"]
        ), "Wrong shape 1 of the result"
        assert (
            result.shape[2] == parameters["M"]
        ), "Wrong shape 2 of the result"
        return result

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = (
            self.nn.laplacian(x) * self.amplitude(x)
            + self.nn(x) * self.amplitude.laplacian(x)
            + 2
            * torch.sum(
                self.nn.gradient(x) * self.amplitude.gradient(x), axis=1
            )
        )
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        assert (
            result.shape[1] == parameters["M"]
        ), "Wrong shape 1 of the result"
        return result

    def weight(self, x: torch.Tensor) -> torch.Tensor:
        """Weight function"""
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = (
            1
            / (parameters["std"] * float(np.sqrt(2 * np.pi)))
            ** parameters["D"]
            * torch.exp(
                -0.5 * torch.sum(x * x, axis=1) / parameters["std"] ** 2
            )
        )
        assert result.shape[0] == x.shape[0], "Wrong shape 0 of the result"
        return result

    def l2Norm(self, x: torch.Tensor) -> torch.Tensor:
        """Computes L2-norm of the trial function"""
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.zeros(parameters["M"]).to(device)
        f = self.forward(x)
        w = self.weight(x)
        for m in range(parameters["M"]):
            result[m] = torch.mean(abs(f[:, m]) ** 2 / w)
        assert (
            result.shape[0] == parameters["M"]
        ), "Wrong shape 0 of the result"
        return result

    def spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Finds current spectrum"""
        global parameters
        global device
        assert x.shape[0] > 0, "Wrong shape 0 of the tensor x"
        assert x.shape[1] == parameters["D"], "Wrong shape 1 of the tensor x"
        result = torch.zeros(parameters["M"]).to(device)
        f = self.forward(x)
        lap = self.laplacian(x)
        w = self.weight(x)
        v = 0.5 * torch.sum(x * x, axis=1)
        for m in range(parameters["M"]):
            result[m] = torch.mean(
                f[:, m] * (-0.5 * lap[:, m] + v * f[:, m]) / w
            )
        result = result / self.l2Norm(x)
        assert (
            result.shape[0] == parameters["M"]
        ), "Wrong shape 0 of the result"
        return result

    def save(self):
        """Saves model weights and biases"""
        torch.save(
            self.state_dict(),
            "../models/" + parameters["configuration"] + ".pt",
        )

    def load(self):
        """Loads model weights and biases"""
        self.load_state_dict(
            "../models/" + parameters["configuration"] + ".pt"
        )


def training(
    trialFunction: TrialFunction,  # Trial function we will train
    steps: int,  # Number of steps
):
    """Trains trial function"""
    global parameters
    global device
    print("Model configuration:")
    print(parameters["configuration"])
    print("")
    print("Training has been started...")
    print(
        "Step, Time [s], Residual-term, Normalization-term,"
        + " Orthogonalization-term, Sorted energies,"
        + "wR, wA, wB, wE"
    )

    time0 = time.time()
    time_history = np.full((steps), np.nan)
    loss_history = np.full((steps), np.nan)
    r_history = np.full((steps), np.nan)
    a_history = np.full((steps), np.nan)
    b_history = np.full((steps), np.nan)
    spectrum_history = np.full((parameters["M"], steps), np.nan)
    wR_history = np.full((steps), np.nan)
    wA_history = np.full((steps), np.nan)
    wB_history = np.full((steps), np.nan)
    wE_history = np.full((steps), np.nan)
    lfweights = {"wR": 1, "wA": 1, "wB": 1, "wE": 1}  # Loss function weights
    optimizer = torch.optim.AdamW(
        params=list(trialFunction.parameters()),
        lr=parameters["learningRate"],
        weight_decay=parameters["weightDecay"],
    )
    lossFunction = nn.MSELoss()
    for step in range(steps):
        # Sample
        x = torch.normal(
            mean=0,
            std=parameters["std"],
            size=(parameters["B"], parameters["D"]),
        ).to(device)
        # Spectrum (Energies)
        e = trialFunction.spectrum(x)
        # Forward (trial function)
        f = trialFunction(x)
        # Weght function
        w = trialFunction.weight(x)
        # Residual
        res = (
            -0.5 * trialFunction.laplacian(x)
            + 0.5 * torch.mul(f.t(), torch.sum(x**2, axis=1)).t()
            - torch.mul(f, e)
        )
        # Squared norm
        sqrNorm = trialFunction.l2Norm(x)
        # A-term
        a = torch.sum((sqrNorm - torch.tensor(1).to(device)) ** 2)
        # B-term
        b = torch.zeros(parameters["M"], parameters["M"]).to(device)
        for m1 in range(1, parameters["M"]):
            for m2 in range(0, m1):
                maxs = (
                    torch.max(f[:, m1]) * torch.max(f[:, m2])
                ).detach() + 0.001
                torch.Tensor.detach(maxs)
                b[m1, m2] = (
                    torch.mean(f[:, m1] * f[:, m2] / w / maxs)
                    ** 2
                    # * torch.mean(f[:, m1] ** 2 / w)
                    # * torch.mean(f[:, m2] ** 2 / w)
                )
        # The total loss
        loss = lossFunction(
            # parameters["wR"]
            # * torch.mean(torch.mean((res) ** 2, axis=0) / sqrNorm)
            # + parameters["wA"] * a
            # + parameters["wB"] * torch.sum(b)
            # + parameters["wE"] * torch.sum(e),
            lfweights["wR"]
            * torch.mean(torch.mean((res) ** 2, axis=0) / sqrNorm.detach())
            + lfweights["wA"] * a
            + lfweights["wB"] * torch.sum(b)
            + lfweights["wE"] * torch.sum(e),
            torch.tensor(0.0).to(device),
        )
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Optimize loss function weights
        if step % parameters["wEpoch"] == 0:
            lfweights["wR"] = (
                min(
                    1
                    / torch.mean(
                        torch.mean((res) ** 2, axis=0) / sqrNorm.detach()
                    )
                    .cpu()
                    .detach()
                    .numpy(),
                    parameters["wRMax"],
                )
                / 4
            )
            lfweights["wA"] = (
                min(1 / a.cpu().detach().numpy(), parameters["wAMax"]) / 4
            )
            lfweights["wB"] = (
                min(
                    1 / torch.sum(b).cpu().detach().numpy(),
                    parameters["wBMax"],
                )
                / 4
            )
            lfweights["wE"] = (
                min(
                    1 / torch.sum(e).cpu().detach().numpy(),
                    parameters["wEMax"],
                )
                / 4
            )
        # Save a state of the model
        time_ = time.time() - time0
        time_history[step] = time_
        loss_history[step] = loss.item()
        r_history[step] = (
            torch.mean(torch.mean((res) ** 2, axis=0) / sqrNorm)
            .cpu()
            .detach()
            .numpy()
        )
        a_history[step] = a.cpu().detach().numpy()
        b_history[step] = torch.sum(b).cpu().detach().numpy()
        spectrum_history[:, step] = e.cpu().detach().numpy()
        wR_history[step] = lfweights["wR"]
        wA_history[step] = lfweights["wA"]
        wB_history[step] = lfweights["wB"]
        wE_history[step] = lfweights["wE"]
        # Release some memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Print the state of the model
        if step % 100 == 0:
            print(
                f"{step}, {time_:.2f}, {r_history[step]:.4e},"
                + f" {a_history[step]:.4e}, {b_history[step]:.4e},"
                + f" {np.sort(spectrum_history[:, step])},"
                + f" {lfweights['wR']:.4e}, {lfweights['wA']:.4e},"
                + f" {lfweights['wB']:.4e}, {lfweights['wE']:.4e}"
            )
    # Save history
    d = {
        "Time[s]": time_history,
        "R-term": r_history,
        "A-term": a_history,
        "B-term": b_history,
        "wR": wR_history,
        "wA": wA_history,
        "wB": wB_history,
        "wE": wE_history,
    }
    for i in range(parameters["M"]):
        d[f"e{i}"] = spectrum_history[i, :]
    history_df = pd.DataFrame(data=d)
    now = dt.datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
    history_df.to_csv(
        "../data/" + dt_string + parameters["configuration"] + ".csv"
    )
    return history_df
