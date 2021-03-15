from collections import defaultdict
from random import Random
from typing import List, Dict

from ffxivcrafter.qlearning.trainer import Environment, QModel, StateType, ActionType, ActionOutcome, train
from ffxivcrafter.environment.state import CraftState, CraftParameter, initial_state, CraftResult, PlayerParameter, \
    ItemParameter, StatusCondition
from ffxivcrafter.environment.action import CraftAction, all_actions, _calc_quality, _calc_progress, Observe, BasicTouch
from ffxivcrafter.simulator import Policy, run_process


def _speed_reward(parameter: CraftParameter, state: CraftState) -> float:
    if state.result == CraftResult.SUCCESS:
        if state.quality < parameter.item.max_quality:
            reward = state.quality - parameter.item.max_quality
        else:
            reward = state.quality / max(1, state.turn)
    elif state.result == CraftResult.FAILED:
        reward = (state.progress - parameter.item.max_progress) - parameter.item.max_quality
    else:
        reward = 0.
    return float(reward)


def _normal_reward(parameter: CraftParameter, state: CraftState) -> float:
    if state.result == CraftResult.SUCCESS:
        reward = state.quality
    else:
        reward = 0.
    return float(reward)


def terminal_reward(parameter: CraftParameter, state: CraftState) -> float:
    return _normal_reward(parameter, state)


def early_reward(parameter: CraftParameter, state: CraftState, prev_state: CraftState) -> float:
    reward = state.quality - prev_state.quality
    reward += state.progress - prev_state.progress
    if state.result == CraftResult.FAILED:
        reward -= state.quality
        reward -= parameter.item.max_progress
    elif state.result == CraftResult.SUCCESS:
        reward -= parameter.item.max_progress
    return float(reward)


class CraftEnvironment(Environment[CraftState, CraftAction]):
    def __init__(self, parameter: CraftParameter, random_state: int = None):
        self.parameter = parameter
        self.random = Random(random_state)

    def initial_state(self) -> StateType:
        return initial_state(self.parameter, self.random.randint(0, 2 ** 60))

    def possible_actions(self, state: CraftState) -> List[CraftAction]:
        return [action for action in all_actions() if action.is_playable(state)]

    def do_action(self, state: CraftState, action: CraftAction) -> List[ActionOutcome]:
        outcomes = []
        for next_state, proba in action.play(self.parameter, state):
            reward = early_reward(self.parameter, next_state, state)
            outcomes.append(ActionOutcome(next_state, reward, proba))
        return outcomes


class QModelPolicy(Policy):
    def __init__(self, model: QModel):
        self.model = model

    def determine(self, parameter: CraftParameter, state: CraftState) -> CraftAction:
        actions = [action for action in all_actions() if action.is_playable(state)]
        assert len(actions) != 0
        return max(actions, key=lambda action: self.model.score(state, action))


class TableQModel(QModel[CraftState, CraftAction]):
    def __init__(self, parameter: CraftParameter, learning_rate: float, random_state: int = None):
        self.parameter = parameter
        self.learning_rate = learning_rate
        self.random = Random(random_state)
        self.table = defaultdict(lambda: self.random.randint(0, parameter.item.max_quality))

    def _get_key(self, state: CraftState, action: CraftAction):
        return state.durability, state.progress // 1000, state.quality // 1000, state.inner_quiet, state.condition, action.ja_name

    def update(self, state: CraftState, action: CraftAction, td_error: float):
        self.table[self._get_key(state, action)] += self.learning_rate * td_error

    def score(self, state: CraftState, action: CraftAction) -> float:
        return self.table[self._get_key(state, action)]


class LinearQModel(QModel[CraftState, CraftAction]):
    def __init__(self, parameter: CraftParameter, learning_rate: float, initial_weights: Dict[str, List[float]] = None, random_state: int = None):
        self.parameter = parameter
        self.learning_rate = learning_rate
        self.random = Random(random_state)
        self.weights = initial_weights or dict()

    def _feature_vector(self, state: CraftState, action: CraftAction) -> List[float]:
        xs = []
        xs.append(state.durability / self.parameter.item.max_durability)
        xs.append(1 if state.durability - action.get_durability_cost(state) <= 0 else 0)
        xs.append(state.progress / self.parameter.item.max_progress)
        xs.append(state.quality / self.parameter.item.max_quality)
        xs.append(1 if state.quality >= self.parameter.item.max_quality else 0)
        xs.append(state.cp / self.parameter.player.max_cp)
        xs.append(1 if state.inner_quiet > 0 else 0)
        xs.append(state.inner_quiet / 10)
        xs.append(1 if state.innovation > 0 else 0)
        xs.append(state.innovation / 10)
        xs.append(1 if state.veneration > 0 else 0)
        xs.append(state.veneration / 10)
        xs.append(1 if state.muscle_memory > 0 else 0)
        xs.append(1 if state.waste_not > 0 else 0)
        xs.append(state.waste_not / 10)
        xs.append(1 if state.great_strides > 0 else 0)
        xs.append(1 if state.final_appraisal > 0 else 0)
        xs.append(1 if state.manipulation > 0 else 0)
        xs.append(state.manipulation / 10)
        xs.append(1 if state.prev_action is None else 0)
        xs.append(1 if type(state.prev_action) is Observe else 0)
        xs.append(1 if type(state.prev_action) is BasicTouch else 0)
        xs.append(_calc_progress(self.parameter, state, 100) / self.parameter.item.max_progress)
        xs.append(_calc_quality(self.parameter, state, 100) / self.parameter.item.max_quality)
        for condition in StatusCondition:
            if condition != StatusCondition.NORMAL:
                xs.append(1 if state.condition == condition else 0)
        xs.append(1.)  # offset
        fxs = [float(x) for x in xs]
        sqs = [x * x for x in xs]
        sqs.pop()
        return fxs + sqs

    def update(self, state: CraftState, action: CraftAction, td_error: float):
        xs = self._feature_vector(state, action)
        if action.ja_name not in self.weights:
            self.weights[action.ja_name] = [self.random.gauss(mu=0, sigma=1) for x in xs]
        ws = self.weights[action.ja_name]
        for i in range(len(ws)):
            ws[i] += self.learning_rate * td_error * xs[i]

    def score(self, state: CraftState, action: CraftAction) -> float:
        xs = self._feature_vector(state, action)
        if action.ja_name not in self.weights:
            self.weights[action.ja_name] = [self.random.gauss(mu=0, sigma=1) for x in xs]
        ws = self.weights[action.ja_name]
        return sum(x * w for x, w in zip(xs, ws))


def get_parameter() -> CraftParameter:
    from ffxivcrafter.environment import factors
    player = PlayerParameter(80, 2867, 2727, 554)
    # special meal for the second restoration
    # item_level = 480
    # durability = 60
    # progress = 9181
    # quality = 64862
    # is_expert_recipe = True
    # shark oil for the last tool improvement
    item_level = 490
    durability = 60
    progress = 10049
    quality = 76939
    is_expert_recipe = True
    # # coffee cookie
    # item_level = 418
    # durability = 80
    # progress = 3705
    # quality = 16582
    # is_expert_recipe = False
    suggested_craftsmanship = factors.suggested_craftsmanship_map[item_level]
    suggested_control = factors.suggested_control_map[item_level]
    item = ItemParameter(item_level, durability, progress, quality, suggested_craftsmanship, suggested_control, is_expert_recipe)
    return CraftParameter(player, item)


def train_model(learning_rate: float, epsilon: float, gamma: float, train_iter: int = 10000) -> QModel:
    parameter = get_parameter()
    env = CraftEnvironment(parameter)
    model = TableQModel(parameter, learning_rate)
    train(env, model, gamma, epsilon, train_iter)
    return model


def eval_model(model: QModel, eval_iter: int = 1) -> float:
    parameter = get_parameter()
    policy = QModelPolicy(model)

    reward_sum = 0
    for _ in range(eval_iter):
        final_state = run_process(policy, parameter, initial_state(parameter), verbose=False)
        reward_sum += terminal_reward(parameter, final_state)
    # if type(model) == TableQModel:
    #     with open("table.json", "w") as f:
    #         import json
    #         keys = []
    #         for key in model.table.keys():
    #             keys.append(list(key))
    #         json.dump(keys, f)
    #     print("table size:", len(model.table))
    return reward_sum / eval_iter


def main():
    parameter = get_parameter()
    env = CraftEnvironment(parameter)
    model = LinearQModel(parameter, learning_rate=0.1)

    print("initial score:", eval_model(model, eval_iter=100))
    print("sample trials from initial model")
    initial_policy = QModelPolicy(model)
    for trial in range(10):
        print(f"trial {trial}:")
        run_process(initial_policy, parameter, initial_state(parameter), verbose=True)
        print("")

    best_score, best_weights = -1e100, None
    try:
        for _ in range(1000):
            train(env, model, gamma=1., epsilon=0.05, n_iter=500)
            score = eval_model(model, eval_iter=100)
            print(f"score: {score}, weights: {model.weights}")
            if score >= best_score:
                best_score, best_weights = score, model.weights
    finally:
        print(best_score)
        best_model = LinearQModel(parameter, learning_rate=0.3, initial_weights=best_weights)
        print("score:", eval_model(best_model, eval_iter=100))
        print("sample trials from best model")
        best_policy = QModelPolicy(best_model)
        for trial in range(10):
            print(f"trial {trial}:")
            run_process(best_policy, parameter, initial_state(parameter), verbose=True)
            print("")


if __name__ == '__main__':
    main()
