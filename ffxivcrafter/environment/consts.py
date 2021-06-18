from ffxivcrafter.environment.state import CraftParameter, PlayerParameter, ItemParameter
from ffxivcrafter.environment import factors


def lv80_player() -> PlayerParameter:
    return PlayerParameter(80, 2867, 2727, 554)


def special_meal_for_second_restoration() -> ItemParameter:
    item_level = 480
    durability = 60
    progress = 9181
    quality = 64862
    is_expert_recipe = True
    suggested_craftsmanship = factors.suggested_craftsmanship_map[item_level]
    suggested_control = factors.suggested_control_map[item_level]
    return ItemParameter(item_level, durability, progress, quality, suggested_craftsmanship, suggested_control, is_expert_recipe)


def coffee_cookie() -> ItemParameter:
    # coffee cookie
    item_level = 418
    durability = 80
    progress = 3705
    quality = 16582
    is_expert_recipe = False
    suggested_craftsmanship = factors.suggested_craftsmanship_map[item_level]
    suggested_control = factors.suggested_control_map[item_level]
    return ItemParameter(item_level, durability, progress, quality, suggested_craftsmanship, suggested_control, is_expert_recipe)

