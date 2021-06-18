import random

from ffxivcrafter.environment.action import all_actions, CraftAction, BasicSynthesis, BasicTouch, MastersMend, \
    ByregotBlessing, RapidSynthesis
from ffxivcrafter.environment.state import CraftParameter, CraftState, CraftResult


class PlayoutStrategy:
    def playout(self, params: CraftParameter, state: CraftState) -> CraftState:
        raise NotImplementedError()


class PureRandom(PlayoutStrategy):
    def playout(self, params: CraftParameter, state: CraftState) -> CraftState:
        while state.result == CraftResult.ONGOING:
            actions = [action for action in all_actions() if action.is_playable(state)]
            action: CraftAction = random.choice(actions)
            next_states, weights = zip(*action.play(params, state))
            state = random.choices(next_states, weights)[0]
        return state


class Greedy(PlayoutStrategy):
    def __init__(self, randomness: float = 0.05):
        self.randomness = randomness

    def playout(self, params: CraftParameter, state: CraftState) -> CraftState:
        rapid_synthesis = RapidSynthesis()  # sagyou
        basic_touch = BasicTouch()  # kakou
        masters_mend = MastersMend()
        bierugo = ByregotBlessing()
        while state.result == CraftResult.ONGOING:
            synthesis_playable = rapid_synthesis.apply(params, state)[1][0].durability > 0
            touch_playable = basic_touch.is_playable(state) and basic_touch.apply(params, state)[0][0].durability > 0
            mend_playable = masters_mend.is_playable(state)
            if not any([synthesis_playable, touch_playable, mend_playable]):
                # pure random
                actions = [action for action in all_actions() if action.is_playable(state)]
                action: CraftAction = random.choice(actions)
            elif not any([synthesis_playable, touch_playable]):
                action = masters_mend
            elif random.random() < self.randomness:
                # pure random
                actions = [action for action in all_actions() if action.is_playable(state)]
                action: CraftAction = random.choice(actions)
            elif synthesis_playable and rapid_synthesis.apply(params, state)[1][0].progress < params.item.max_progress:
                action = rapid_synthesis
            elif touch_playable:
                action = basic_touch
            else:
                action = rapid_synthesis
            next_states, weights = zip(*action.play(params, state))
            state = random.choices(next_states, weights)[0]
        return state