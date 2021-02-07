from random import Random
from enum import Enum

from ffxivcrafter.environment.util import clip
from typing import NewType, Optional


class StatusCondition(Enum):
    NORMAL = "通常"
    GOOD = "高品質"
    EXCELLENT = "最高品質"
    POOR = "低品質"
    CENTRED = "安定"  # success rate +25% (50% -> 75%)
    PLIANT = "高能率"  # cp consumption is halfed. rounded up. 7 -> 4
    STURDY = "頑丈"  # duration consumption is halfed. rounded up. 5 -> 3
    MALLEABLE = "高進捗"  # progress x1.5
    PRIMED = "長持続"  # buff +2 turn

    def __init__(self, ja_name: str):
        self.ja_name = ja_name

    def __str__(self):
        return self.ja_name


class CraftResult(Enum):
    ONGOING = 0
    FAILED = 1
    SUCCESS = 2


RawLevel = NewType("RawLevel", int)
InternalLevel = NewType("InternalLevel", int)
Durability = NewType("Durability", int)
Progress = NewType("Progress", int)
Quality = NewType("Quality", int)
CP = NewType("CP", int)


class PlayerParameter:
    def __init__(self, raw_level: RawLevel, craftsmanship: int, control: int, max_cp: CP):
        self.raw_level = raw_level
        self.craftsmanship = craftsmanship
        self.control = control
        self.max_cp = max_cp

    def __str__(self):
        return f"<Player raw_level: {self.raw_level}, craftsmanship: {self.craftsmanship}, control: {self.control}, max_cp: {self.max_cp}>"


class ItemParameter:
    def __init__(self, internal_level: InternalLevel, max_durability: Durability, max_progress: Progress, max_quality: Quality, standard_craftsmanship: int, standard_control: int, expert_recipe: bool):
        self.internal_level = internal_level
        self.max_durability = max_durability
        self.max_progress = max_progress
        self.max_quality = max_quality
        self.standard_craftsmanship = standard_craftsmanship
        self.standard_control = standard_control
        self.expert_recipe = expert_recipe

    def __str__(self):
        return f"<Item internal_level: {self.internal_level}, max_durability: {self.max_durability}, max_progress: {self.max_progress}, max_quality: {self.max_quality}, " \
               f"standard_craftsmanship: {self.standard_craftsmanship}, standard_control: {self.standard_control}, expert_recipe: {self.expert_recipe}>"


class CraftParameter:
    def __init__(self, player: PlayerParameter, item: ItemParameter):
        self.player = player
        self.item = item

    def __str__(self):
        return f"<Parameter {self.player}, {self.item}>"


class CraftState:
    def __init__(self, durability: Durability, progress: Progress, quality: Quality, cp: CP, condition: StatusCondition,
                 inner_quiet: int, innovation: int, veneration: int, muscle_memory: int, waste_not: int,
                 great_strides: int, final_appraisal: int, manipulation: int, turn: int,
                 prev_action: Optional["CraftAction"], random_state: Random, result: CraftResult):
        self.durability = durability
        self.progress = progress
        self.quality = quality
        self.cp = cp
        self.condition = condition

        self.inner_quiet = inner_quiet
        self.innovation = innovation
        self.veneration = veneration
        self.muscle_memory = muscle_memory
        self.waste_not = waste_not
        self.great_strides = great_strides
        self.final_appraisal = final_appraisal
        self.manipulation = manipulation
        self.turn = turn

        self.prev_action = prev_action
        self.random_state = random_state
        self.result = result

    def clip(self, parameter: CraftParameter):
        self.durability = clip(0, self.durability, parameter.item.max_durability)
        self.progress= clip(0, self.progress, parameter.item.max_progress)
        self.quality = clip(0, self.quality, parameter.item.max_quality)
        self.cp = clip(0, self.cp, parameter.player.max_cp)

        self.inner_quiet = clip(0, self.inner_quiet, 11)
        self.innovation = max(0, self.innovation)
        self.veneration = max(0, self.veneration)
        self.muscle_memory = max(0, self.muscle_memory)
        self.waste_not = max(0, self.waste_not)
        self.great_strides = max(0, self.great_strides)
        self.final_appraisal = max(0, self.final_appraisal)
        self.manipulation = max(0, self.manipulation)
        self.turn = max(0, self.turn)

    def __copy__(self):
        return CraftState(
            self.durability,
            self.progress,
            self.quality,
            self.cp,
            self.condition,
            self.inner_quiet,
            self.innovation,
            self.veneration,
            self.muscle_memory,
            self.waste_not,
            self.great_strides,
            self.final_appraisal,
            self.manipulation,
            self.turn,
            self.prev_action,
            Random(self.random_state.getstate()),
            self.result
        )

    def __str__(self):
        return f"<State durability: {self.durability}, progress: {self.progress}, quality: {self.quality}, " \
               f"cp: {self.cp}, condition: {self.condition}, " \
               f"inner_quiet: {self.inner_quiet}, innovation: {self.innovation}, veneration: {self.veneration}, " \
               f"muscle_memory: {self.muscle_memory}, waste_not: {self.waste_not}, " \
               f"great_strides: {self.great_strides}, final_appraisal: {self.final_appraisal}, " \
               f"manipulation: {self.manipulation}, turn: {self.turn}, " \
               f"prev_action: {self.prev_action}, random_state: {self.random_state}, result: {self.result}>"


def initial_state(parameter: CraftParameter, random_state: int = None):
    return CraftState(
        parameter.item.max_durability,
        0,
        0,
        parameter.player.max_cp,
        StatusCondition.NORMAL,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        Random(random_state),
        CraftResult.ONGOING
    )