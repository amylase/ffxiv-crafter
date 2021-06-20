from ffxivcrafter.environment.state import CraftParameter, CraftState, CraftResult, StatusCondition
from ffxivcrafter.environment.factors import get_control_factor, get_craftsmanship_factor, get_transition_probabilities
from typing import List, Tuple
from copy import copy
from math import floor


ProbabilisticState = Tuple[CraftState, float]


def deterministic(state: CraftState) -> List[ProbabilisticState]:
    return [(state, 1.)]


def tick(parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
    """
    progress turn
    """
    assert(state.result == CraftResult.ONGOING)
    assert state.cp >= 0
    state = copy(state)
    state.clip(parameter)

    if type(state.prev_action) is FinalAppraisal:
        return deterministic(state)

    if state.progress >= parameter.item.max_progress:
        if state.final_appraisal <= 0:
            state.result = CraftResult.SUCCESS
            return deterministic(state)
        else:
            state.final_appraisal = 0
            state.progress = parameter.item.max_progress - 1
    if state.durability <= 0:
        state.result = CraftResult.FAILED
        return deterministic(state)
    # implement buff tick here
    state.innovation -= 1
    state.veneration -= 1
    state.muscle_memory -= 1
    state.waste_not -= 1
    state.great_strides -= 1
    state.final_appraisal -= 1
    if state.prev_action.consume_muscle_memory:
        state.muscle_memory = 0
    if state.prev_action.consume_great_strides:
        state.great_strides = 0
    if state.manipulation > 0 and type(state.prev_action) is not Manipulation:
        state.durability += 5
    state.manipulation -= 1
    state.turn += 1
    state.clip(parameter)

    # implement state transition here
    next_probas = get_transition_probabilities(parameter, state)
    next_states = []
    for next_condition, probability in next_probas.items():
        next_state = copy(state)
        next_state.condition = next_condition
        next_states.append((next_state, probability))
    # return it!
    return next_states


class CraftAction:
    def __init__(self, ja_name: str, cp_cost: int, durability_cost: int, consume_muscle_memory: bool, consume_great_strides: bool):
        self.ja_name = ja_name
        self._cp_cost = cp_cost
        self._durability_cost = durability_cost
        self.consume_muscle_memory = consume_muscle_memory
        self.consume_great_strides = consume_great_strides

    def get_cp_cost(self, state: CraftState) -> int:
        if state.condition == StatusCondition.PLIANT:
            return (self._cp_cost + 1) // 2
        else:
            return self._cp_cost

    def get_durability_cost(self, state: CraftState) -> int:
        cost = self._durability_cost
        if state.waste_not > 0:
            cost = (cost + 1) // 2
        if state.condition == StatusCondition.STURDY:
            cost = (cost + 1) // 2
        return cost

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        raise NotImplementedError()

    def play(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        result_states = []
        for action_result, prob1 in self.apply(parameter, state):
            for next_state, prob2 in tick(parameter, action_result):
                result_states.append((next_state, prob1 * prob2))
        return result_states

    def is_playable(self, state: CraftState) -> bool:
        return state.result == CraftResult.ONGOING and self.get_cp_cost(state) <= state.cp

    def __str__(self):
        return self.ja_name

    def __eq__(self, other):
        if not isinstance(other, CraftAction):
            return False
        return self.ja_name == other.ja_name

    def __hash__(self):
        return hash(self.ja_name)


def all_actions() -> List[CraftAction]:
    return [cls() for cls in CraftAction.__subclasses__()]


def _calc_progress(parameter: CraftParameter, state: CraftState, efficiency: int) -> int:
    craftsmanship = parameter.player.craftsmanship
    raw_level = parameter.player.raw_level
    item = parameter.item
    # implement buff here
    if state.veneration > 0:
        efficiency *= 1.5
    if state.muscle_memory > 0:
        efficiency *= 2
    progress = floor(efficiency / 100 * (0.21 * craftsmanship + 2) * (10000 + craftsmanship) / (
                    10000 + item.standard_craftsmanship) * get_craftsmanship_factor(raw_level, item.internal_level))
    if state.condition == StatusCondition.MALLEABLE:
        progress *= 1.5
    return floor(progress)


def _calc_quality(parameter: CraftParameter, state: CraftState, efficiency: int) -> int:
    control = parameter.player.control
    raw_level = parameter.player.raw_level
    item = parameter.item
    # implement buff here
    efficiency_coefficient = 1
    if state.innovation > 0:
        efficiency_coefficient += 0.5
    if state.great_strides > 0:
        efficiency_coefficient += 1
    efficiency *= efficiency_coefficient
    inner_quiet_ratio = max(1., 1. + (state.inner_quiet - 1) * 0.2)
    control *= inner_quiet_ratio
    quality = floor(efficiency / 100 * (0.35 * control + 35) * (10000 + control) / (10000 + item.standard_control) * get_control_factor(raw_level, item.internal_level))
    if state.condition == StatusCondition.POOR:
        quality *= 0.5
    elif state.condition == StatusCondition.GOOD:
        quality *= 1.5
    elif state.condition == StatusCondition.EXCELLENT:
        quality *= 4.
    return floor(quality)


def _calc_buff_turns(base: int, state: CraftState) -> int:
    return (base + 2) if state.condition == StatusCondition.PRIMED else base


class BasicSynthesis(CraftAction):
    def __init__(self):
        super(BasicSynthesis, self).__init__("作業", 0, 10, True, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.progress += _calc_progress(parameter, state, 120)  # todo: set 100 when crafter level is low
        new_state.prev_action = self
        return deterministic(new_state)


class BasicTouch(CraftAction):
    def __init__(self):
        super(BasicTouch, self).__init__("加工", 18, 10, False, True)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.quality += _calc_quality(parameter, state, 100)
        if new_state.inner_quiet > 0:
            new_state.inner_quiet += 1
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class MastersMend(CraftAction):
    def __init__(self):
        super(MastersMend, self).__init__("マスターズメンド", 88, 0, False, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability += 30
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)
    

class InnerQuiet(CraftAction):
    def __init__(self):
        super(InnerQuiet, self).__init__("インナークワイエット", 18, 0, False, False)

    def is_playable(self, state: CraftState) -> bool:
        return super(InnerQuiet, self).is_playable(state) and state.inner_quiet == 0

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.inner_quiet += 1
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class DelicateSynthesis(CraftAction):
    def __init__(self):
        super(DelicateSynthesis, self).__init__("精密作業", 32, 10, True, True)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.progress += _calc_progress(parameter, state, 100)
        new_state.quality += _calc_quality(parameter, state, 100)
        if state.inner_quiet > 0:
            new_state.inner_quiet += 1
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class CarefulSynthesis(CraftAction):
    def __init__(self):
        super(CarefulSynthesis, self).__init__("模範作業", 7, 10, True, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.progress += _calc_progress(parameter, state, 150)
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class Groundwork(CraftAction):
    def __init__(self):
        super(Groundwork, self).__init__("下地作業", 18, 20, True, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        if state.durability < self.get_durability_cost(state):
            new_state.durability = 0
            new_state.progress += _calc_progress(parameter, state, 150)
        else:
            new_state.durability -= self.get_durability_cost(state)
            new_state.progress += _calc_progress(parameter, state, 300)
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class Observe(CraftAction):
    def __init__(self):
        super(Observe, self).__init__("経過観察", 7, 0, False, True)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class ByregotBlessing(CraftAction):
    def __init__(self):
        super(ByregotBlessing, self).__init__("ビエルゴの祝福", 24, 10, False, True)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        efficiency = max(100, 20 * (state.inner_quiet - 1) + 100)
        new_state.quality += _calc_quality(parameter, state, efficiency)
        new_state.inner_quiet = 0
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class PreparatoryTouch(CraftAction):
    def __init__(self):
        super(PreparatoryTouch, self).__init__("下地加工", 40, 20, False, True)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.quality += _calc_quality(parameter, state, 200)
        if new_state.inner_quiet > 0:
            new_state.inner_quiet += 2
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class RapidSynthesis(CraftAction):
    def __init__(self):
        super(RapidSynthesis, self).__init__("突貫作業", 0, 10, True, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.prev_action = self
        failure_state = copy(new_state)
        success_state = copy(new_state)
        success_state.progress += _calc_progress(parameter, state, 500)
        success_proba = 0.75 if state.condition == StatusCondition.CENTRED else 0.5
        new_states = [(failure_state, 1. - success_proba), (success_state, success_proba)]
        return new_states


class IntensiveSynthesis(CraftAction):
    def __init__(self):
        super(IntensiveSynthesis, self).__init__("集中作業", 6, 10, True, False)

    def is_playable(self, state: CraftState) -> bool:
        return state.condition in [StatusCondition.GOOD, StatusCondition.EXCELLENT] and super(IntensiveSynthesis, self).is_playable(state)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.cp -= self.get_cp_cost(state)
        new_state.progress += _calc_progress(parameter, state, 400)
        new_state.prev_action = self
        return deterministic(new_state)


class HastyTouch(CraftAction):
    def __init__(self):
        super(HastyTouch, self).__init__("ヘイスティタッチ", 0, 10, False, True)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.prev_action = self
        failure_state = copy(new_state)
        success_state = copy(new_state)
        success_state.quality += _calc_quality(parameter, state, 100)
        if state.inner_quiet > 0:
            success_state.inner_quiet += 1
        success_proba = 0.85 if state.condition == StatusCondition.CENTRED else 0.6
        new_states = [(failure_state, 1. - success_proba), (success_state, success_proba)]
        return new_states


class PreciseTouch(CraftAction):
    def __init__(self):
        super(PreciseTouch, self).__init__("集中加工", 18, 10, False, True)

    def is_playable(self, state: CraftState) -> bool:
        return state.condition in [StatusCondition.GOOD, StatusCondition.EXCELLENT] and super(PreciseTouch, self).is_playable(state)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.prev_action = self
        new_state.cp -= self.get_cp_cost(state)
        new_state.quality += _calc_quality(parameter, state, 150)
        if state.inner_quiet > 0:
            new_state.inner_quiet += 2
        return deterministic(new_state)


class PatientTouch(CraftAction):
    def __init__(self):
        super(PatientTouch, self).__init__("専心加工", 6, 10, False, True)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        failure_state = copy(new_state)
        failure_state.inner_quiet = (state.inner_quiet + 1) // 2
        success_state = copy(new_state)
        success_state.quality += _calc_quality(parameter, state, 100)
        success_state.inner_quiet *= 2
        success_proba = 0.75 if state.condition == StatusCondition.CENTRED else 0.5
        new_states = [(failure_state, 1. - success_proba), (success_state, success_proba)]
        return new_states


class TricksOfTheTrade(CraftAction):
    def __init__(self):
        super(TricksOfTheTrade, self).__init__("秘訣", 0, 0, False, False)

    def is_playable(self, state: CraftState) -> bool:
        return state.condition in [StatusCondition.GOOD, StatusCondition.EXCELLENT] and super(TricksOfTheTrade, self).is_playable(state)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.cp += 20
        new_state.prev_action = self
        return deterministic(new_state)


class Innovation(CraftAction):
    def __init__(self):
        super(Innovation, self).__init__("イノベーション", 18, 0, False, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.innovation = _calc_buff_turns(5, state)  # = 4 + 1
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class Veneration(CraftAction):
    def __init__(self):
        super(Veneration, self).__init__("ヴェネレーション", 18, 0, False, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.veneration = _calc_buff_turns(5, state)  # = 4 + 1
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class MuscleMemory(CraftAction):
    def __init__(self):
        super(MuscleMemory, self).__init__("確信", 6, 10, False, False)

    def is_playable(self, state: CraftState) -> bool:
        return state.prev_action is None and super(MuscleMemory, self).is_playable(state)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.prev_action = self
        new_state.cp -= self.get_cp_cost(state)
        new_state.progress += _calc_progress(parameter, state, 300)
        new_state.muscle_memory = _calc_buff_turns(6, state)  # = 5 + 1
        return deterministic(new_state)


class FocusedSynthesis(CraftAction):
    def __init__(self):
        super(FocusedSynthesis, self).__init__("注視作業", 5, 10, True, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.prev_action = self
        new_state.cp -= self.get_cp_cost(state)
        success_state = copy(new_state)
        new_state.progress += _calc_progress(parameter, state, 200)
        failure_state = copy(new_state)
        if type(state.prev_action) is Observe:
            return deterministic(success_state)
        else:
            success_proba = 0.75 if state.condition == StatusCondition.CENTRED else 0.5
            return [(success_state, success_proba), (failure_state, 1. - success_proba)]


class StandardTouch(CraftAction):
    def __init__(self):
        super(StandardTouch, self).__init__("中級加工", 32, 10, False, True)

    def get_cp_cost(self, state: CraftState) -> int:
        return 18 if type(state.prev_action) is BasicTouch else 32

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.prev_action = self
        new_state.cp -= self.get_cp_cost(state)
        new_state.quality += _calc_quality(parameter, state, 125)
        if state.inner_quiet > 0:
            new_state.inner_quiet += 1
        return deterministic(new_state)


class FocusedTouch(CraftAction):
    def __init__(self):
        super(FocusedTouch, self).__init__("注視加工", 18, 10, False, True)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.prev_action = self
        new_state.cp -= self.get_cp_cost(state)
        if state.inner_quiet > 0:
            new_state.inner_quiet += 1
        success_state = copy(new_state)
        new_state.quality += _calc_quality(parameter, state, 150)
        failure_state = copy(new_state)
        if type(state.prev_action) is Observe:
            return deterministic(success_state)
        else:
            success_proba = 0.75 if state.condition == StatusCondition.CENTRED else 0.5
            return [(success_state, success_proba), (failure_state, 1. - success_proba)]


class Reflect(CraftAction):
    def __init__(self):
        super(Reflect, self).__init__("真価", 24, 10, False, True)

    def is_playable(self, state: CraftState) -> bool:
        return state.prev_action is None and super(Reflect, self).is_playable(state)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.prev_action = self
        new_state.cp -= self.get_cp_cost(state)
        new_state.quality += _calc_quality(parameter, state, 100)
        new_state.inner_quiet = 3
        return deterministic(new_state)


class WasteNot(CraftAction):
    def __init__(self):
        super(WasteNot, self).__init__("倹約", 56, 0, False, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.prev_action = self
        new_state.cp -= self.get_cp_cost(state)
        new_state.waste_not = _calc_buff_turns(5, state)  # = 4 + 1
        return deterministic(new_state)


class WasteNotII(CraftAction):
    def __init__(self):
        super(WasteNotII, self).__init__("長期倹約", 98, 0, False, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.prev_action = self
        new_state.cp -= self.get_cp_cost(state)
        new_state.waste_not = _calc_buff_turns(9, state)  # = 8 + 1
        return deterministic(new_state)


class PrudentTouch(CraftAction):
    def __init__(self):
        super(PrudentTouch, self).__init__("倹約加工", 25, 5, False, True)

    def is_playable(self, state: CraftState) -> bool:
        return state.waste_not <= 0 and super(PrudentTouch, self).is_playable(state)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.durability -= self.get_durability_cost(state)
        new_state.quality += _calc_quality(parameter, state, 100)
        if new_state.inner_quiet > 0:
            new_state.inner_quiet += 1
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        return deterministic(new_state)


class GreatStrides(CraftAction):
    def __init__(self):
        super(GreatStrides, self).__init__("グレートストライド", 32, 0, False, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        new_state.great_strides = _calc_buff_turns(4, state)  # = 3 + 1
        return deterministic(new_state)


class FinalAppraisal(CraftAction):
    def __init__(self):
        super(FinalAppraisal, self).__init__("最終確認", 1, 0, False, False)

    def is_playable(self, state: CraftState) -> bool:
        # forbid continuous final appraisal because it makes episodes longer
        return type(state.prev_action) != FinalAppraisal and super(FinalAppraisal, self).is_playable(state)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        new_state.final_appraisal = _calc_buff_turns(5, state)  # no need to add 1 because the next tick is skipped.
        return deterministic(new_state)


class Manipulation(CraftAction):
    def __init__(self):
        super(Manipulation, self).__init__("マニピュレーション", 96, 0, False, False)

    def apply(self, parameter: CraftParameter, state: CraftState) -> List[ProbabilisticState]:
        new_state = copy(state)
        new_state.cp -= self.get_cp_cost(state)
        new_state.prev_action = self
        new_state.manipulation = _calc_buff_turns(9, state)  # 8 + 1
        return deterministic(new_state)