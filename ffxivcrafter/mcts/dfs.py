from functools import lru_cache
from typing import Tuple, Optional

from ffxivcrafter.environment.action import all_actions, CraftAction
from ffxivcrafter.environment.state import CraftParameter, CraftState, CraftResult
from ffxivcrafter.mcts.playout import Greedy


def evaluate(params: CraftParameter, state: CraftState) -> float:
    greedy_algo = Greedy(randomness=0, force_normal=True)
    final_state = greedy_algo.playout(params, state)
    if final_state.result != CraftResult.SUCCESS:
        return 0.
    return final_state.quality / params.item.max_quality


def dfs(params: CraftParameter, state: CraftState, depth: int) -> Tuple[Optional[CraftAction], float]:
    if depth == 0 or state.result != CraftResult.ONGOING:
        return None, evaluate(params, state)
    actions = [action for action in all_actions() if action.is_playable(state)]
    expectations = []
    best_expectation = 0.
    for action in actions:
        expectation = 0.
        upper_bound = 1.
        for next_state, proba in sorted(action.play(params, state), key=lambda tup: tup[1], reverse=True):
            sub_exp = dfs(params, next_state, depth - 1)[1]
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

    from ffxivcrafter.environment.consts import coffee_cookie, lv80_player, special_meal_for_second_restoration, special_meal_for_fourth_restoration
    from ffxivcrafter.environment.state import initial_state as get_initial_state
    player = lv80_player()
    item = special_meal_for_fourth_restoration()
    params = CraftParameter(player, item)
    state = get_initial_state(params)
    rng = random.Random()
    print(item)

    depth = 2
    rng.seed(0)
    total_time = -time.time()
    while state.result == CraftResult.ONGOING:
        elapsed = -time.time()
        print(state)
        action = dfs(params, state, depth)[0]
        elapsed += time.time()
        print(f"{action}, elapsed: {elapsed:.3f}")
        next_states, weights = zip(*action.play(params, state))
        state = rng.choices(next_states, weights)[0]
        if elapsed < 0.4:
            print(f"increasing depth: {depth} -> {depth + 1}")
            depth += 1
    print("done")
    print(state)
    print(state.result)
    total_time += time.time()
    print(f"total time: {total_time:.3f}")