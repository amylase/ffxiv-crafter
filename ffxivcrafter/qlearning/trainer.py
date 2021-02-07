from typing import Generic, TypeVar, List
from random import Random
import math


StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")


class QModel(Generic[StateType, ActionType]):
    def update(self, state: StateType, action: ActionType, td_error: float):
        raise NotImplementedError()

    def score(self, state: StateType, action: ActionType) -> float:
        raise NotImplementedError()


class ActionOutcome(Generic[StateType]):
    def __init__(self, next_state: StateType, reward: float, probability: float):
        self.next_state = next_state
        self.reward = reward
        self.probability = probability


class Environment(Generic[StateType, ActionType]):
    def initial_state(self) -> StateType:
        raise NotImplementedError()

    def possible_actions(self, state: StateType) -> List[ActionType]:
        raise NotImplementedError()

    def do_action(self, state: StateType, action: ActionType) -> List[ActionOutcome]:
        raise NotImplementedError()


def train(env: Environment, model: QModel, gamma: float, epsilon: float, n_iter: int, random_state: int = None):
    random = Random(random_state)
    for _iter in range(n_iter):
        state = env.initial_state()
        while True:
            candidate_actions = env.possible_actions(state)
            if len(candidate_actions) == 0:
                break
            # determine the next action by epsilon-greedy policy
            if random.random() < epsilon:
                action = random.choice(candidate_actions)
            else:
                action = max(candidate_actions, key=lambda action: model.score(state, action))

            outcomes = env.do_action(state, action)
            outcome_weights = [outcome.probability for outcome in outcomes]
            sampled_outcome = random.choices(outcomes, outcome_weights)[0]  # type: ActionOutcome
            next_state = sampled_outcome.next_state
            reward = sampled_outcome.reward

            next_actions = env.possible_actions(next_state)
            best_next_action_score = max(map(lambda action: model.score(next_state, action), next_actions)) if next_actions else 0.

            td_error = reward + gamma * best_next_action_score - model.score(state, action)
            model.update(state, action, td_error)
            state = next_state
