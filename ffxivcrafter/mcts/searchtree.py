import math
import random
from typing import List, Dict

from ffxivcrafter.environment.action import CraftAction, all_actions, BasicSynthesis
from ffxivcrafter.environment.state import CraftState, CraftParameter, CraftResult
from ffxivcrafter.mcts.playout import PlayoutStrategy


class TreeNode:
    def __init__(self, reward_sum: float = 0., sample_count: float = 0.):
        self.reward_sum = reward_sum
        self.sample_count = sample_count


class PlayableStateNode(TreeNode):
    def __init__(self, state: CraftState, next_ids: Dict[CraftAction, int] = None):
        super(PlayableStateNode, self).__init__()
        self.state = state
        self.next_ids = next_ids

    def is_expanded(self):
        return self.next_ids is not None


class ProbabilisticStateNode(TreeNode):
    def __init__(self, next_ids: List[int], weights: List[float]):
        super(ProbabilisticStateNode, self).__init__()
        assert len(next_ids) == len(weights)
        assert len(next_ids) > 0
        self.next_ids = next_ids
        self.weights = weights


class SearchTree:
    def __init__(self, parameter: CraftParameter, initial_state: CraftState, playout_algo: PlayoutStrategy):
        self.parameter = parameter
        self.playable_nodes: List[PlayableStateNode] = []
        self.probabilistic_nodes: List[ProbabilisticStateNode] = []
        self.root = self._add_playable_node(initial_state)
        self.playout_algo = playout_algo

    def _add_playable_node(self, state: CraftState) -> int:
        node_id = len(self.playable_nodes)
        self.playable_nodes.append(PlayableStateNode(state))
        return node_id

    def _add_probabilistic_node(self, state: CraftState, action: CraftAction) -> int:
        next_ids = []
        weights = []
        for next_state, probability in action.play(self.parameter, state):
            next_id = self._add_playable_node(next_state)
            next_ids.append(next_id)
            weights.append(probability)
        node_id = len(self.probabilistic_nodes)
        self.probabilistic_nodes.append(ProbabilisticStateNode(next_ids, weights))
        return node_id

    def _expand_playable_node(self, playable_node_id: int):
        node = self.playable_nodes[playable_node_id]
        # todo: ignore obviously weak move
        actions = [action for action in all_actions() if action.is_playable(node.state)]
        next_ids = dict()
        for action in actions:
            next_id = self._add_probabilistic_node(node.state, action)
            next_ids[action] = next_id
        node.next_ids = next_ids

    def _playout(self, state: CraftState) -> float:
        # returns reward of playout.
        state = self.playout_algo.playout(self.parameter, state)
        # todo: adjust reward
        if state.result == CraftResult.FAILED:
            return 0.
        return state.quality / self.parameter.item.max_quality

    def _search(self, playable_node_id: int) -> float:
        node = self.playable_nodes[playable_node_id]
        if not node.is_expanded():
            self._expand_playable_node(playable_node_id)
        tried_actions, fresh_actions = [], []
        for action, probabilistic_node_id in node.next_ids.items():
            probabilistic_node = self.probabilistic_nodes[probabilistic_node_id]
            if probabilistic_node.sample_count == 0.:
                fresh_actions.append(action)
            else:
                tried_actions.append(action)
        if not tried_actions and not fresh_actions:
            # no actions; state is terminal. returns the immediate reward by calling playout
            reward = self._playout(node.state)
        else:
            if not fresh_actions:
                # all actions already explored at least once. select the next node
                # by solving the multiarmed bandit problem
                # todo: use algorithm for best arm identification (ucb1 is for regret minimization)
                t = 0
                for action in tried_actions:
                    n = self.probabilistic_nodes[node.next_ids[action]]
                    t += n.sample_count
                def ucb1(action: CraftAction) -> float:
                    n = self.probabilistic_nodes[node.next_ids[action]]
                    s = n.sample_count
                    w = n.reward_sum
                    return (w / s) + ((2 * math.log(t) / s) ** 0.5)
                action = max(tried_actions, key=ucb1)
            else:
                # there is at least one node which is not explored yet.
                action = random.choice(fresh_actions)
            next_probabilistic_node_id = node.next_ids[action]
            probabilistic_node = self.probabilistic_nodes[next_probabilistic_node_id]
            next_playable_node_id = random.choices(probabilistic_node.next_ids, probabilistic_node.weights)[0]
            if not fresh_actions:
                reward = self._search(next_playable_node_id)
            else:
                reward = self._playout(self.playable_nodes[next_playable_node_id].state)
            probabilistic_node.sample_count += 1.
            probabilistic_node.reward_sum += reward
        node.sample_count += 1.
        node.reward_sum += reward
        return reward

    def search(self, node_id: int = None):
        node_id = node_id or self.root
        self._search(node_id)

    def best_next_action(self) -> CraftAction:
        node = self.playable_nodes[self.root]
        assert node.is_expanded()

        def evaluate(action: CraftAction) -> float:
            next_node = self.probabilistic_nodes[node.next_ids[action]]
            return next_node.sample_count
        return max(node.next_ids.keys(), key=evaluate)

    def shift(self, action: CraftAction, next_state: CraftState):
        assert self.playable_nodes[self.root].is_expanded()
        probabilistic_node_id = self.playable_nodes[self.root].next_ids[action]
        probabilistic_node = self.probabilistic_nodes[probabilistic_node_id]
        for next_playable_node_id in probabilistic_node.next_ids:
            node = self.playable_nodes[next_playable_node_id]
            if node.state == next_state:
                self.root = next_playable_node_id
                return
        assert False, "this must not happen"


if __name__ == '__main__':
    from ffxivcrafter.environment.consts import coffee_cookie, lv80_player, special_meal_for_second_restoration
    from ffxivcrafter.environment.state import initial_state as get_initial_state
    from ffxivcrafter.mcts.playout import PureRandom, Greedy
    player = lv80_player()
    item = special_meal_for_second_restoration()
    params = CraftParameter(player, item)
    state = get_initial_state(params)
    playout = Greedy()
    tree = SearchTree(params, state, playout)
    print(item)
    while state.result == CraftResult.ONGOING:
        print(state)
        for _ in range(20000):
            tree.search()
        action = tree.best_next_action()
        print(action)
        next_states, weights = zip(*action.play(params, state))
        state = random.choices(next_states, weights)[0]
        # tree.shift(action, state)
        tree = SearchTree(params, state, playout)
    print("done")
    print(state)
    print(state.result)