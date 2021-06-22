import random
from typing import List, Tuple

from ffxivcrafter.environment.action import all_actions
from ffxivcrafter.environment.state import CraftParameter, CraftState, CraftResult, initial_state, StatusCondition
from ffxivcrafter.mcts.dfs import EvaluatorType, dfs, terminal_score


def calc_final_state(params: CraftParameter, state: CraftState, evaluator: EvaluatorType, depth: int) -> CraftState:
    # note: the result depends on the rng state
    while state.result == CraftResult.ONGOING:
        action, score = dfs(params, state, evaluator, depth)
        next_states, weights = zip(*action.play(params, state))
        state = random.choices(next_states, weights)[0]
    return state


def generate_samples(params: CraftParameter, evaluator: EvaluatorType, depth: int) -> List[Tuple[CraftState, float]]:
    state = initial_state(params)
    samples = []
    while state.result == CraftResult.ONGOING:
        actions = [action for action in all_actions() if action.is_playable(state)]
        action_scores = []
        for action in actions:
            expected_score = 0.
            next_states = action.play(params, state)
            for next_state, proba in next_states:
                if next_state.result != CraftResult.ONGOING:
                    continue
                final_state = calc_final_state(params, next_state, evaluator, depth)
                score = terminal_score(params, final_state)
                samples.append((final_state, score))
                expected_score += score * proba
            action_scores.append((action, expected_score))
        next_action = max(action_scores, key=lambda tup: tup[1])[0]
        next_states, weights = zip(*next_action.play(params, state))
        state = random.choices(next_states, weights)[0]
    return samples


def _generate(args):
    random.seed()
    return generate_samples(*args)


if __name__ == '__main__':
    import time

    from ffxivcrafter.environment.consts import coffee_cookie, lv80_player, special_meal_for_second_restoration, special_meal_for_fourth_restoration
    from ffxivcrafter.mcts.dfs import greedy_evaluator

    player = lv80_player()
    item = special_meal_for_second_restoration()
    params = CraftParameter(player, item)
    evaluator = greedy_evaluator
    depth = 2

    import multiprocessing
    cpus = 8
    pool = multiprocessing.Pool(processes=cpus)

    elapsed = -time.time()
    samples = pool.map(_generate, [(params, evaluator, depth)] * cpus)
    elapsed += time.time()
    samples = sum(samples, [])
    print(f"obtained {len(samples)} samples in {elapsed:.3f} secs.")

    import pickle
    from pathlib import Path
    filepath = Path("C:\\Users\\amyla\\PycharmProjects\\ffxiv-crafter\\data\\samples.pkl")
    with filepath.open("wb") as f:
        pickle.dump(samples, f)