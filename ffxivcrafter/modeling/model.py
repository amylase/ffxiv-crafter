from typing import List

from ffxivcrafter.environment.state import CraftState, StatusCondition


def extract_feature(state: CraftState) -> List[float]:
    base_features = [
        state.durability,
        state.progress,
        state.quality,
        state.cp,
        state.inner_quiet,
        state.manipulation,
        state.waste_not,
        state.innovation,
        state.veneration,
        state.great_strides,
        state.muscle_memory,
        state.cp >= 24,
        state.inner_quiet > 0,
        state.veneration > 0,
        state.condition == StatusCondition.GOOD,
        state.condition == StatusCondition.CENTRED,
        state.condition == StatusCondition.MALLEABLE,
        state.condition == StatusCondition.PLIANT,
        state.condition == StatusCondition.PRIMED,
        state.condition == StatusCondition.STURDY,
    ]
    v = [float(x) for x in base_features]
    sq = []
    for i, x in enumerate(v):
        for y in v[:i + 1]:
            sq.append(x * y)
    # return v
    return v + sq


def score_to_target(score: float) -> float:
    return score


def target_to_score(target: float) -> float:
    return target


class LinearEvaluator:
    def __init__(self, weights: List[float], intercept: float):
        self.weights = weights
        self.intercept = intercept

    def evaluate(self, state: CraftState) -> float:
        target = sum(weight * feature for weight, feature in zip(self.weights, extract_feature(state))) + self.intercept
        return target_to_score(target)
