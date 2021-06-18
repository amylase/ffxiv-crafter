import random

from ffxivcrafter.environment.action import BasicSynthesis, BasicTouch, Reflect
from ffxivcrafter.environment.parameters import COFFEE_COOKIE, PATCH_5_4_CRAFTER
from ffxivcrafter.environment.state import CraftParameter, CraftState
from ffxivcrafter.evaluator import success_reward, quality_reward
from ffxivcrafter.simulator import Policy, iterate_processes


class HeuristicsPolicy(Policy):
    def __init__(self, random_state=None):
        self.random = random.Random(random_state)

    def determine(self, parameter: CraftParameter, state: CraftState):
        if state.prev_action is None:
            return Reflect()

        return self.random.choice([action for action in [BasicSynthesis(), BasicTouch()] if action.is_playable(state)])


def main():
    item = COFFEE_COOKIE
    player = PATCH_5_4_CRAFTER
    params = CraftParameter(player, item)
    policy = HeuristicsPolicy()
    n_iter = 1000
    final_states = iterate_processes(policy, params, n_iter)

    named_aggregators = [
        ("success rate", success_reward),
        ("quality", quality_reward)
    ]
    for name, aggregator in named_aggregators:
        average = sum(aggregator(params, state) for state in final_states) / n_iter
        print(name, average)


if __name__ == '__main__':
    main()