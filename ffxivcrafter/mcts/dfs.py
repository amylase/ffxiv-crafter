import pickle
from functools import lru_cache
from typing import Tuple, Optional, Callable

from ffxivcrafter.environment.action import all_actions, CraftAction, RapidSynthesis, BasicTouch, DelicateSynthesis, \
    ByregotBlessing, PreparatoryTouch, HastyTouch, PreciseTouch, PatientTouch, Innovation, StandardTouch, FocusedTouch, \
    GreatStrides, InnerQuiet, FinalAppraisal, Veneration, Manipulation, WasteNot, WasteNotII
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


def is_completable(params: CraftParameter, state: CraftState) -> bool:
    rapid_synth = RapidSynthesis()
    if not rapid_synth.is_playable(state):
        return False
    for next_state, proba in rapid_synth.play(params, state):
        if next_state.result == CraftResult.SUCCESS:
            return True
    return False


_quality_actions = frozenset([BasicTouch(), ByregotBlessing(), PreparatoryTouch(), HastyTouch(), PreciseTouch(),
                              PatientTouch(), Innovation(), StandardTouch(), FocusedTouch(), GreatStrides()])
def is_quality_action(action: CraftAction) -> bool:
    return action in _quality_actions


def is_meaningful_action(action: CraftAction, params: CraftParameter, state: CraftState):
    if isinstance(action, InnerQuiet):
        return state.inner_quiet <= 0
    elif isinstance(action, FinalAppraisal):
        return state.final_appraisal <= 0
    elif isinstance(action, BasicTouch):
        return not isinstance(state.prev_action, BasicTouch)
    elif isinstance(action, PatientTouch):
        return state.inner_quiet <= 8
    elif isinstance(action, RapidSynthesis):
        for next_state, proba in action.apply(params, state):
            if next_state.progress >= params.item.max_progress:
                return False
        return True
    elif isinstance(action, Veneration):
        duration = 6 if state.condition == StatusCondition.PRIMED else 4
        return duration > state.veneration
    elif isinstance(action, Innovation):
        duration = 6 if state.condition == StatusCondition.PRIMED else 4
        return duration > state.innovation
    elif isinstance(action, Manipulation):
        duration = 10 if state.condition == StatusCondition.PRIMED else 8
        return duration > state.manipulation
    elif isinstance(action, WasteNot):
        duration = 6 if state.condition == StatusCondition.PRIMED else 4
        return duration > state.waste_not
    elif isinstance(action, WasteNotII):
        duration = 10 if state.condition == StatusCondition.PRIMED else 8
        return duration > state.waste_not
    return True


def dfs(params: CraftParameter, state: CraftState, evaluator: EvaluatorType, depth: int) -> Tuple[Optional[CraftAction], float]:
    if state.result != CraftResult.ONGOING:
        return None, terminal_score(params, state)
    if depth == 0:
        return None, evaluator(params, state)
    is_completable_state = is_completable(params, state)
    actions = [action for action in all_actions() if action.is_playable(state) and (is_completable_state or not is_quality_action(action))]
    expectations = []
    best_expectation = 0.
    for action in actions:
        if not is_meaningful_action(action, params, state):
            continue
        expectation = 0.
        upper_bound = 1.
        # ignore special next states if action is not Final Appraisal
        if action.ja_name == "最終確認":
            next_condition = state.condition
        else:
            next_condition = StatusCondition.NORMAL
        next_states = sorted(action.play(params, state), key=lambda tup: tup[1], reverse=True)
        next_normal_states = [(next_state, proba) for next_state, proba in next_states
                              if next_state.condition == next_condition and next_state.result == CraftResult.ONGOING]
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

    from ffxivcrafter.environment.consts import coffee_cookie, lv80_player, lv80_player_with_buffs, special_meal_for_second_restoration, special_meal_for_fourth_restoration
    from ffxivcrafter.environment.state import initial_state as get_initial_state
    player = lv80_player_with_buffs()
    item = special_meal_for_fourth_restoration()
    params = CraftParameter(player, item)
    state = get_initial_state(params)
    rng = random.Random()
    print(item)

    depth = 4
    evaluator = greedy_evaluator
    # with open("../../data/model_params.json", "rb") as f:
    #     model_params = json.load(f)
    #     model = LinearEvaluator(**model_params)
    #
    #     def evaluator(params: CraftParameter, state: CraftState) -> float:
    #         return model.evaluate(state)
    rng.seed(10000)
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