import random

from ffxivcrafter.environment.action import all_actions, CraftAction, BasicSynthesis, BasicTouch, MastersMend, \
    ByregotBlessing, RapidSynthesis, GreatStrides, Manipulation, StandardTouch, FocusedTouch, Innovation
from ffxivcrafter.environment.state import CraftParameter, CraftState, CraftResult, StatusCondition


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
    def __init__(self, randomness: float = 0.05, force_normal: bool = False):
        self.randomness = randomness
        self.force_normal = force_normal

    def playout(self, params: CraftParameter, state: CraftState) -> CraftState:
        _state = state
        basic_synthesis = BasicSynthesis()  # sagyou
        basic_touch = BasicTouch()  # kakou
        standard_touch = StandardTouch()  # chukyu kakou
        masters_mend = MastersMend()
        bierugo = ByregotBlessing()
        great_strides = GreatStrides()
        while state.result == CraftResult.ONGOING:
            synthesis_playable = basic_synthesis.apply(params, state)[0][0].durability > 0
            touch_playable = basic_touch.is_playable(state) and basic_touch.apply(params, state)[0][0].durability > 0
            mend_playable = masters_mend.is_playable(state)
            bierugo_playable = bierugo.is_playable(state) and bierugo.apply(params, state)[0][0].durability > 0
            great_strides_playable = great_strides.is_playable(state)
            if not any([synthesis_playable, touch_playable, mend_playable]):
                # pure random
                actions = [action for action in all_actions() if action.is_playable(state)]
                action = actions[0] if self.randomness <= 0. else random.choice(actions)
            elif not any([synthesis_playable, touch_playable]):
                action = masters_mend
            elif random.random() < self.randomness:
                # pure random
                actions = [action for action in all_actions() if action.is_playable(state)]
                action: CraftAction = random.choice(actions)
            # elif great_strides_playable and state.cp < 74 and state.great_strides == 0:
            #     action = great_strides
            elif synthesis_playable and state.progress < basic_synthesis.play(params, state)[0][0].progress < params.item.max_progress:
                action = basic_synthesis
            # elif bierugo_playable and state.cp < 42:
            #     action = bierugo
            elif touch_playable:
                if state.prev_action is not None and state.prev_action.ja_name == "加工":
                    action = standard_touch
                elif state.prev_action is not None and state.prev_action.ja_name == "経過観察":
                    action = FocusedTouch()
                elif state.innovation == 0:
                    action = Innovation()
                else:
                    action = basic_touch
            else:
                action = basic_synthesis
            if self.force_normal:
                state = action.play(params, state)[0][0]
                state.condition = StatusCondition.NORMAL
            else:
                next_states, weights = zip(*action.play(params, state))
                state: CraftState = random.choices(next_states, weights)[0]
        return state