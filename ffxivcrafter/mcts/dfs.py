from functools import lru_cache

from ffxivcrafter.environment.action import all_actions
from ffxivcrafter.environment.state import CraftParameter, CraftState, CraftResult
from ffxivcrafter.mcts.playout import Greedy


@lru_cache(maxsize=None)
def evaluate(params: CraftParameter, state: CraftState) -> float:
    greedy_algo = Greedy(randomness=0)
    final_state = greedy_algo.playout(params, state)
    if final_state.result != CraftResult.SUCCESS:
        return 0.
    return final_state.quality / params.item.max_quality


@lru_cache(maxsize=None)
def dfs(params: CraftParameter, state: CraftState, depth: int) -> float:
    if depth == 0 or state.result != CraftResult.ONGOING:
        return evaluate(params, state)
    actions = [action for action in all_actions() if action.is_playable(state)]
    expectations = []
    for action in actions:
        expectation = 0.
        for next_state, proba in action.play(params, state):
            expectation += dfs(params, next_state, depth - 1) * proba
        expectations.append(expectation)
    return max(expectations)


if __name__ == '__main__':
    import random
    import time

    from ffxivcrafter.environment.consts import coffee_cookie, lv80_player, special_meal_for_second_restoration
    from ffxivcrafter.environment.state import initial_state as get_initial_state
    player = lv80_player()
    item = special_meal_for_second_restoration()
    params = CraftParameter(player, item)
    state = get_initial_state(params)
    print(item)
    depth = 1
    while state.result == CraftResult.ONGOING:
        elapsed = -time.time()
        print(state)
        actions = [action for action in all_actions() if action.is_playable(state)]
        expectations = []
        for action in actions:
            expectation = 0.
            for next_state, proba in action.play(params, state):
                expectation += dfs(params, next_state, depth) * proba
            expectations.append((action, expectation))
        action = max(expectations, key=lambda tup: tup[1])[0]
        print(action)
        next_states, weights = zip(*action.play(params, state))
        state = random.choices(next_states, weights)[0]
        elapsed += time.time()
        if elapsed < 1:
            print(f"increasing depth: {depth} -> {depth + 1}")
            depth += 1
    print("done")
    print(state)
    print(state.result)