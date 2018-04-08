from machine import TsetlinMachine
from data import noisy_XOR, split_train_validate, MNIST


def noisy_XOR_experiment():
    tsetlin_machine = TsetlinMachine(number_clauses=20,
                                     number_action_states=100,
                                     precision=3.9,
                                     threshold=15)

    X, y = noisy_XOR(5000, flip_fraction=0.4)
    val_X, val_y = noisy_XOR(5000, flip_fraction=0.)

    tsetlin_machine.fit(X, y, val_X, val_y, 200)
    print('Final training accuracy:', tsetlin_machine.accuracy(X, y), ' (this should be around 0.6)')
    print('Final validation accuracy:', tsetlin_machine.accuracy(val_X, val_y), ' (according to the paper this should be 0.994)')

def MNIST_experiment():
    """I tried it with binary inputs but (at least with the current implementation) it is way too slow"""
    tsetlin_machine = TsetlinMachine(number_clauses=1000,
                                     number_action_states=1000,
                                     precision=3.0,
                                     threshold=10)

    X, y, val_X, val_y = MNIST()

    tsetlin_machine.fit(X, y, val_X, val_y, 300)
    print('Final training accuracy:', tsetlin_machine.accuracy(X, y))
    print('Final validation accuracy:', tsetlin_machine.accuracy(val_X, val_y))
