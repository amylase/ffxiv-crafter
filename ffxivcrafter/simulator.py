from ffxivcrafter.environment.state import PlayerParameter, ItemParameter, CraftParameter, initial_state, CraftResult, CraftState
from ffxivcrafter.environment.action import all_actions, CraftAction
import random


class Policy:
    def determine(self, parameter: CraftParameter, state: CraftState) -> CraftAction:
        raise NotImplementedError()


class RandomPolicy(Policy):
    def __init__(self, random_state=None):
        self.random = random.Random(random_state)

    def determine(self, parameter: CraftParameter, state: CraftState):
        return self.random.choice([action for action in all_actions() if action.is_playable(state)])


class SmartPolicy(Policy):
    def __init__(self, random_state=None):
        self.random = random.Random(random_state)

    def determine(self, parameter: CraftParameter, state: CraftState):
        from ffxivcrafter.environment.action import MastersMend
        if state.cp >= 88 and parameter.item.max_durability - state.durability >= 30:
            return MastersMend()
        return self.random.choice([action for action in all_actions() if action.is_playable(state) and type(action) != MastersMend])


def run_process(policy: Policy, parameter: CraftParameter, state: CraftState, verbose: bool=True) -> CraftState:
    while state.result == CraftResult.ONGOING:
        if verbose: print(state)
        action = policy.determine(parameter, state)
        if verbose: print(action)
        new_states_and_probas = action.play(parameter, state)
        new_states, new_probas = zip(*new_states_and_probas)
        state = random.choices(new_states, new_probas)[0]
    if verbose: print(state)
    return state


def average_quality(policy: Policy, parameter: CraftParameter, n_iter: int) -> float:
    quality_sum = 0
    for _ in range(n_iter):
        final_state = run_process(policy, parameter, initial_state(parameter), verbose=False)
        if final_state.result == CraftResult.SUCCESS:
            quality_sum += final_state.quality
    return quality_sum / n_iter


def main():
    from ffxivcrafter.environment import factors
    player = PlayerParameter(80, 2867, 2727, 554)
    # coffee cookie
    item_level = 480
    durability = 60
    progress = 9181
    quality = 64862
    is_expert_recipe = True
    suggested_craftsmanship = factors.suggested_craftsmanship_map[item_level]
    suggested_control = factors.suggested_control_map[item_level]
    item = ItemParameter(item_level, durability, progress, quality, suggested_craftsmanship, suggested_control, is_expert_recipe)
    parameter = CraftParameter(player, item)
    print(parameter)

    print(average_quality(RandomPolicy(), parameter, 1000))
    print(average_quality(SmartPolicy(), parameter, 1000))


if __name__ == '__main__':
    main()