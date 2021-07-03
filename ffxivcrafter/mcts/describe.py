import pickle

from ffxivcrafter.environment.state import CraftParameter
from ffxivcrafter.mcts.dfs import dfs
from ffxivcrafter.environment.consts import special_meal_for_fourth_restoration, lv80_player_with_buffs
from ffxivcrafter.mcts.dfs import greedy_evaluator


def main():
    params = CraftParameter(lv80_player_with_buffs(), special_meal_for_fourth_restoration())
    with open("../../data/turn2.pkl", "rb") as f:
        state = pickle.load(f)
    print(state)
    print(greedy_evaluator(params, state) * params.item.max_quality)
    # action, score = dfs(params, state, greedy_evaluator, 3)
    # print(action, score * params.item.max_quality)


if __name__ == '__main__':
    main()