import numpy as np
import random


class Automaton:
    def __init__(self, number_action_states):
        """Initialize Automaton, number_action_states is the number of states in one action, random initialization of state"""
        self.number_action_states = number_action_states
        self.state = np.random.randint(2*number_action_states)

    def evaluate(self):
        """Whether Automaton currently evaluates to True"""
        return self.state >= self.number_action_states

    def reward(self):
        """Reward the Automaton, pushing it further to the boundaries"""
        if self.state >= self.number_action_states:
            self.state = min(2 * self.number_action_states - 1, self.state + 1)
        else:
            self.state = max(0, self.state - 1)

    def penalize(self):
        """Penalize the Automaton, pushing it towards the middle of the state space"""
        if self.state >= self.number_action_states:
            self.state = self.state - 1
        else:
            self.state = self.state + 1


class AutomataTeam:
    def __init__(self, number_inputs, number_action_states, positive):
        """Team of Automata that is responsible for one clause. For each input there is an inclusion and an exclusion
           Automata. Positive determines whether the polarity is positive or not"""
        self.number_inputs = number_inputs
        self.positive = positive
        self.polarity = 1 if positive else -1
        self.positive_automata = [Automaton(number_action_states) for _ in range(number_inputs)]
        self.negative_automata = [Automaton(number_action_states) for _ in range(number_inputs)]

    def evaluate(self, X):
        """If the team thinks the element should be positively included and it evaluates to False return False. Reversed for
           negated elements. If X adheres to all the rules return True"""
        for x_element, positive, negative in zip(X, self.positive_automata, self.negative_automata):
            if positive.evaluate() and not x_element:
                return False
            if negative.evaluate() and x_element:
                return False
        return True

    def value(self, X):
        """Value used in summation for voting, the evaluation multiplied with the polarity"""
        return self.evaluate(X) * self.polarity


class TsetlinMachine:
    def __init__(self, number_clauses, number_action_states, precision, threshold):
        """
        The Learning Machine.
        :param number_clauses: Number of clauses or AutomataTeams per output bit.
        :param number_action_states: Number of stats per action
        :param precision: Precision in the algorithm, higher makes it more flexible
        :param threshold: If the output is more extreme than the threshold do not perform stochastic updates
        """
        self.number_clauses = number_clauses
        self.number_action_states = number_action_states
        self.precision = precision
        self.threshold = threshold

    def fit(self, X, y, val_X, val_y, n_epoch=10):
        """
        Fit TsetlinMachine on training set
        :param X: Two dimensional training input array of Sample X Dimensionality
        :param y: Two dimensional training output array of Sample X Dimensionality
        :param X: Two dimensional validation input array of Sample X Dimensionality
        :param y: Two dimensional validation output array of Sample X Dimensionality
        :param n_epoch: Number of passes over training set
        """
        number_inputs = X.shape[1]
        number_outputs = y.shape[1]

        # Initialize all the teams for every output, each of them learning a specific pattern
        self.automata_teams = [[AutomataTeam(number_inputs, self.number_action_states, index % 2 == 0) \
                                for index in range(self.number_clauses)] for _ in range(number_outputs)]

        indices = list(range(X.shape[0]))
        for epoch in range(n_epoch):
            random.shuffle(indices)
            for index, sample_index in enumerate(indices):
                if index % 100 == 0:
                    print(index)
                current_X, current_y = X[sample_index], y[sample_index]
                self.train_row(current_X, current_y)
            print('Epoch:', epoch, ' Validation accuracy:', self.accuracy(val_X, val_y))

    def predict(self, X):
        """Predict two dimensional bit output of y"""
        y = []
        for index in range(X.shape[0]):
            current_X = X[index]
            current_y = []
            for y_element_automata_teams in self.automata_teams:
                current_y.append(np.sum([team.value(current_X) for team in y_element_automata_teams]) >= 0)
            y.append(current_y)
        return np.array(y)

    def accuracy(self, X, y):
        """Accuracy over two dimensions, just the fraction of correct bits on the predicted y and passed y"""
        predicted_y = self.predict(X)
        return np.mean(predicted_y == y)

    def train_row(self, current_X, current_y):
        """
        Apply learning rules to current row
        :param current_X: Input vector
        :param current_y: Output vector
        """
        # Loop over the different output bits and their corresponding automata teams
        for y_element, y_element_automata_teams in zip(current_y, self.automata_teams):
            # Evaluate current function value for current output bit
            y_vote_value = np.sum([team.value(current_X) for team in y_element_automata_teams])
            for automata_team in y_element_automata_teams:
                # If this automata team is positive, which is an odd index in the paper, apply the positive rules
                if automata_team.positive:
                    if y_element:
                        if self.sample_feedback(y_vote_value, True):
                            self.apply_type_1_feedback(current_X, automata_team)
                    else:
                        if self.sample_feedback(y_vote_value, False):
                            self.apply_type_2_feedback(current_X, automata_team)
                # Reversed for negative polarity automata teams
                else:
                    if y_element:
                        if self.sample_feedback(y_vote_value, True):
                            self.apply_type_2_feedback(current_X, automata_team)
                    else:
                        if self.sample_feedback(y_vote_value, False):
                            self.apply_type_1_feedback(current_X, automata_team)


    def sample_feedback(self, y_vote_value, y_true):
        """
        Return whether or not to sample feedback. The more certain the right choice was made, the less likely to apply the rules
        :param y_vote_value: Current decision value for the relevant output bit
        :param y_true: Whether the ground truth y bit is on
        :return: Boolean whether to apply the feedback rules
        """
        if y_true:
            return np.random.rand() < (self.threshold - max(-self.threshold, min(self.threshold, y_vote_value))) / (
                        2 * self.threshold)
        else:
            return np.random.rand() < (self.threshold + max(-self.threshold, min(self.threshold, y_vote_value))) / (
                        2 * self.threshold)

    def apply_type_1_feedback_automaton(self, clause_output, element, automaton):
        """Go over all the options in the matrix in the paper and sample whether or not to reward or penalize the automaton"""
        if clause_output:
            if element:
                if automaton.evaluate():
                    if self.sample_high_probability():
                        automaton.reward()
                else:
                    if self.sample_high_probability():
                        automaton.penalize()
            else:
                if not automaton.evaluate():
                    if self.sample_low_probability():
                        automaton.reward()
        else:
            if automaton.evaluate():
                if self.sample_low_probability():
                    automaton.penalize()
            else:
                if self.sample_low_probability():
                    automaton.reward()


    def apply_type_1_feedback(self, current_X, automata_team):
        """Apply type 1 feedback to the passed automata team for the given X"""
        clause_output = automata_team.evaluate(current_X)
        for x_element, positive_automaton, negative_automaton in zip(current_X,
                                                                     automata_team.positive_automata,
                                                                     automata_team.negative_automata):
            self.apply_type_1_feedback_automaton(clause_output, x_element, positive_automaton)
            self.apply_type_1_feedback_automaton(clause_output, not x_element, negative_automaton)

    def apply_type_2_feedback(self, current_X, automata_team):
        """Apply type 2 feedback to the passed automata team (clause generator) with the given X"""
        clause_output = automata_team.evaluate(current_X)
        for x_element, positive_automaton, negative_automaton in zip(current_X,
                                                                     automata_team.positive_automata,
                                                                     automata_team.negative_automata):
            if clause_output:
                if not x_element:
                    if not positive_automaton.evaluate():
                        positive_automaton.penalize()
                if x_element:
                    if not negative_automaton.evaluate():
                        negative_automaton.penalize()


    def sample_high_probability(self):
        """High probability sampling from the reward/penalty table"""
        return np.random.rand() < (self.precision - 1) / self.precision

    def sample_low_probability(self):
        """Low probability sampling from the reward/penalty table"""
        return np.random.rand() < 1 / self.precision
