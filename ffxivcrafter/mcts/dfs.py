import pickle
from functools import lru_cache
from typing import Tuple, Optional, Callable

from ffxivcrafter.environment.action import all_actions, CraftAction
from ffxivcrafter.environment.state import CraftParameter, CraftState, CraftResult, StatusCondition
from ffxivcrafter.mcts.playout import Greedy
from ffxivcrafter.modeling.model import LinearEvaluator

EvaluatorType = Callable[[CraftParameter, CraftState], float]


def terminal_score(params: CraftParameter, state: CraftState) -> float:
    if state.result != CraftResult.SUCCESS:
        return 0.
    return state.quality / params.item.max_quality


def greedy_evaluator(params: CraftParameter, state: CraftState) -> float:
    greedy_algo = Greedy(randomness=0, force_normal=True)
    final_state = greedy_algo.playout(params, state)
    return terminal_score(params, final_state)


def dfs(params: CraftParameter, state: CraftState, evaluator: EvaluatorType, depth: int) -> Tuple[Optional[CraftAction], float]:
    if state.result != CraftResult.ONGOING:
        return None, terminal_score(params, state)
    if depth == 0:
        return None, evaluator(params, state)
    actions = [action for action in all_actions() if action.is_playable(state)]
    expectations = []
    best_expectation = 0.
    for action in actions:
        expectation = 0.
        upper_bound = 1.
        # ignore special next states
        next_states = sorted(action.play(params, state), key=lambda tup: tup[1], reverse=True)
        next_normal_states = [(next_state, proba) for next_state, proba in next_states
                              if next_state.condition == StatusCondition.NORMAL and next_state.result == CraftResult.ONGOING]
        next_terminal_states = [(next_state, proba) for next_state, proba in next_states
                                if next_state.result != CraftResult.ONGOING]
        terminal_proba = sum(proba for next_state, proba in next_terminal_states)
        ongoing_proba = 1. - terminal_proba
        normal_proba = sum(proba for next_state, proba in next_normal_states)
        next_normal_states = [(next_state, proba * (ongoing_proba / normal_proba)) for next_state, proba in next_normal_states]
        for next_state, proba in next_normal_states + next_terminal_states:
            sub_exp = dfs(params, next_state, evaluator, depth - 1)[1]
            expectation += sub_exp * proba
            upper_bound -= (1 - sub_exp) * proba
            if upper_bound < best_expectation:
                break
        expectations.append((action, expectation))
        best_expectation = max(best_expectation, expectation)
    return max(expectations, key=lambda tup: tup[1])


if __name__ == '__main__':
    import random
    import time
    import json

    from ffxivcrafter.environment.consts import coffee_cookie, lv80_player, special_meal_for_second_restoration, special_meal_for_fourth_restoration
    from ffxivcrafter.environment.state import initial_state as get_initial_state
    player = lv80_player()
    item = special_meal_for_second_restoration()
    params = CraftParameter(player, item)
    state = get_initial_state(params)
    rng = random.Random()
    print(item)

    depth = 3
    # evaluator = greedy_evaluator
    with open("../../data/model_params.json", "rb") as f:
        model_params = json.load(f)
        model = LinearEvaluator(**model_params)

        def evaluator(params: CraftParameter, state: CraftState) -> float:
            return model.evaluate(state)
    rng.seed(1)
    total_time = -time.time()
    while state.result == CraftResult.ONGOING:
        elapsed = -time.time()
        print(state)
        current_score = evaluator(params, state)
        print(f"score: {current_score:.3f} (predicted quality: {int(current_score * params.item.max_quality)})")
        action, score = dfs(params, state, evaluator, depth)
        elapsed += time.time()
        print(f"{action}, elapsed: {elapsed:.3f}, score: {score:.3f} (predicted quality: {int(score * params.item.max_quality)})")
        next_states, weights = zip(*action.play(params, state))
        state = rng.choices(next_states, weights)[0]
        if elapsed < 0.3:
            print(f"increasing depth: {depth} -> {depth + 1}")
            depth += 1
    print("done")
    print(state)
    print(state.result)
    total_time += time.time()
    print(f"total time: {total_time:.3f}")