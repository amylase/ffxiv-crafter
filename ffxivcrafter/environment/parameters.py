from ffxivcrafter.environment import factors
from ffxivcrafter.environment.state import ItemParameter, PlayerParameter


def _calculate_item_parameters(item_level: int, durability: int, progress: int, quality: int, is_expert_recipe: bool) -> ItemParameter:
    suggested_craftsmanship = factors.suggested_craftsmanship_map[item_level]
    suggested_control = factors.suggested_control_map[item_level]
    return ItemParameter(item_level, durability, progress, quality, suggested_craftsmanship, suggested_control, is_expert_recipe)


COFFEE_COOKIE = _calculate_item_parameters(418, 80, 3705, 16582, False)
SECOND_ISHGARDIAN_RESTORATION_ITEM = _calculate_item_parameters(480, 60, 9181, 64862, True)
SKYSTEEL_TOOL_IMPROVEMENT_ITEM = _calculate_item_parameters(490, 60, 10049, 76939, True)

PATCH_5_4_CRAFTER = PlayerParameter(80, 2867, 2727, 554)
