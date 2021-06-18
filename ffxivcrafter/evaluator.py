from ffxivcrafter.environment.state import CraftParameter, CraftState, CraftResult


def speed_reward(parameter: CraftParameter, state: CraftState) -> float:
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


def quality_reward(parameter: CraftParameter, state: CraftState) -> float:
    if state.result == CraftResult.SUCCESS:
        reward = state.quality
    else:
        reward = 0.
    return float(reward)


def terminal_reward(parameter: CraftParameter, state: CraftState) -> float:
    return quality_reward(parameter, state)


def early_reward(parameter: CraftParameter, state: CraftState, prev_state: CraftState) -> float:
    reward = state.quality - prev_state.quality
    reward += state.progress - prev_state.progress
    if state.result == CraftResult.FAILED:
        reward -= state.quality
        reward -= parameter.item.max_progress
    elif state.result == CraftResult.SUCCESS:
        reward -= parameter.item.max_progress
    return float(reward)


def success_reward(parameter: CraftParameter, state: CraftState) -> float:
    return 1. if state.result == CraftResult.SUCCESS else 0.