import pytest
from tsetlin.machine import Automaton, AutomataTeam


@pytest.fixture()
def automaton():
    return Automaton(2)


@pytest.fixture()
def automata_team():
    team = AutomataTeam(2, 2, True)
    first_inclusion_automaton = Automaton(2)
    first_inclusion_automaton.state = 2
    second_inclusion_automaton = Automaton(2)
    second_inclusion_automaton.state = 0
    team.inclusion_automata = [first_inclusion_automaton, second_inclusion_automaton]
    first_exclusion_automaton = Automaton(2)
    first_exclusion_automaton.state = 0
    second_exclusion_automaton = Automaton(2)
    second_exclusion_automaton.state = 2
    team.exclusion_automata = [first_exclusion_automaton, second_exclusion_automaton]
    return team


def test_automaton_reward(automaton):
    automaton.state = 1
    automaton.reward()
    assert automaton.state == 0
    automaton.reward()
    assert automaton.state == 0
    automaton.state = 2
    automaton.reward()
    assert automaton.state == 3
    automaton.reward()
    assert automaton.state == 3


def test_automaton_penalize(automaton):
    automaton.state = 1
    automaton.penalize()
    assert automaton.state == 2
    automaton.penalize()
    assert automaton.state == 1


def test_team_evaluate(automata_team):
    X = [True, False]
    assert automata_team.evaluate(X)
    X = [False, True]
    assert not automata_team.evaluate(X)
