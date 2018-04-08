from machine import TsetlinMachine
from data import noisy_XOR, split_train_validate
from experiments import noisy_XOR_experiment, MNIST_experiment

noisy_XOR_experiment()
#MNIST_experiment()