import copy
import itertools
import logging
import os
import pickle
import time

import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.search import Graph
from overcooked_ai_py.utils import pos_distance, manhattan_distance

from .search import SearchTree
from ..data.planners import load_saved_action_manager, PLANNERS_DIR
from ..mdp.overcooked_env import OvercookedEnv
from ..mdp.overcooked_mdp import (
    OvercookedState,
    OvercookedGridworld,
    PlayerState,
    EVENT_TYPES,
    ObjectState,
)

logger = logging.getLogger(__name__)

# Run planning logic with additional checks and
# computation to prevent or identify possible minor errors
SAFE_RUN = False
LOGUNIT = 500
TRAINNINGUNIT = 5000
MAX_NUM_STATES = 19000


class MotionPlanner:
    """A planner that computes optimal plans for a single agent to
    arrive at goal positions and orientations in an OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
        counter_goals (list): list of positions of counters we will consider
                              as valid motion goals
    """

    def __init__(self, mdp, counter_goals=[]):
        self.mdp = mdp

        # If positions facing counters should be
        # allowed as motion goals
        self.counter_goals = counter_goals

        # Graph problem that solves shortest path problem
        # between any position & orientation start-goal pair
        self.graph_problem = self._graph_from_grid()
        self.motion_goals_for_pos = self._get_goal_dict()

    def get_plan(self, start_pos_and_or, goal_pos_and_or):
        """
        Returns pre-computed plan from initial agent position
        and orientation to a goal position and orientation.

        Args:
            start_pos_and_or (tuple): starting (pos, or) tuple
            goal_pos_and_or (tuple): goal (pos, or) tuple
        """
        action_plan, pos_and_or_path, plan_cost = self._compute_plan(
            start_pos_and_or, goal_pos_and_or)
        return action_plan, pos_and_or_path, plan_cost

    def get_gridworld_distance(self, start_pos_and_or, goal_pos_and_or):
        """Number of actions necessary to go from starting position
        and orientations to goal position and orientation (not including
        interaction action)"""
        assert self.is_valid_motion_start_goal_pair(
            start_pos_and_or, goal_pos_and_or
        ), "Goal position and orientation were not a valid motion goal"
        # Removing interaction cost
        return self.graph_problem.dist(start_pos_and_or, goal_pos_and_or) - 1

    def get_gridworld_pos_distance(self, pos1, pos2):
        """Minimum (over possible orientations) number of actions necessary
        to go from starting position to goal position (not including
        interaction action)."""
        # NOTE: currently unused, pretty bad code. If used in future, clean up
        min_cost = np.Inf
        for d1, d2 in itertools.product(Direction.ALL_DIRECTIONS, repeat=2):
            start = (pos1, d1)
            end = (pos2, d2)
            if self.is_valid_motion_start_goal_pair(start, end):
                plan_cost = self.get_gridworld_distance(start, end)
                if plan_cost < min_cost:
                    min_cost = plan_cost
        return min_cost

    def is_valid_motion_start_goal_pair(self,
                                        start_pos_and_or,
                                        goal_pos_and_or,
                                        debug=False):
        if not self.is_valid_motion_goal(goal_pos_and_or):
            return False
        if not self.positions_are_connected(start_pos_and_or, goal_pos_and_or):
            return False
        return True

    def is_valid_motion_goal(self, goal_pos_and_or):
        """
        Checks that desired single-agent goal state (position and orientation)
        is reachable and is facing a terrain feature
        """
        goal_position, goal_orientation = goal_pos_and_or
        if goal_position not in self.mdp.get_valid_player_positions():
            return False

        ## temp commit since actions should not be limited to only sub-goals
        ## that complete a task, but should include actions such as wait and put
        ## down item and switch sub-goals, which do not always face a terrain
        ## with features.
        # # Restricting goals to be facing a terrain feature
        # pos_of_facing_terrain = Action.move_in_direction(
        #     goal_position, goal_orientation)
        # facing_terrain_type = self.mdp.get_terrain_type_at_pos(
        #     pos_of_facing_terrain)
        # if facing_terrain_type == " " or (facing_terrain_type == "X" and
        #                                   pos_of_facing_terrain
        #                                   not in self.counter_goals):
        #     return False
        return True

    def _compute_plan(self, start_motion_state, goal_motion_state):
        """Computes optimal action plan for single agent movement

        Args:
            start_motion_state (tuple): starting positions and orientations
            goal_motion_state (tuple): goal positions and orientations
        """
        assert self.is_valid_motion_start_goal_pair(start_motion_state,
                                                    goal_motion_state)
        positions_plan = self._get_position_plan_from_graph(
            start_motion_state, goal_motion_state)
        (
            action_plan,
            pos_and_or_path,
            plan_length,
        ) = self.action_plan_from_positions(positions_plan, start_motion_state,
                                            goal_motion_state)
        return action_plan, pos_and_or_path, plan_length

    def positions_are_connected(self, start_pos_and_or, goal_pos_and_or):
        return self.graph_problem.are_in_same_cc(start_pos_and_or,
                                                 goal_pos_and_or)

    def _get_position_plan_from_graph(self, start_node, end_node):
        """Recovers positions to be reached by agent after the start node to
        reach the end node"""
        node_path = self.graph_problem.get_node_path(start_node, end_node)
        assert node_path[0] == start_node and node_path[-1] == end_node
        positions_plan = [state_node[0] for state_node in node_path[1:]]
        return positions_plan

    def action_plan_from_positions(self, position_list, start_motion_state,
                                   goal_motion_state):
        """
        Recovers an action plan reaches the goal motion position and
        orientation, and executes and interact action.

        Args:
            position_list (list): list of positions to be reached after the
                starting position (does not include starting position, but
                includes ending position)
            start_motion_state (tuple): starting position and orientation
            goal_motion_state (tuple): goal position and orientation

        Returns:
            action_plan (list): list of actions to reach goal state
            pos_and_or_path (list): list of (pos, or) pairs visited during plan
                execution (not including start, but including goal)
        """
        goal_position, goal_orientation = goal_motion_state
        action_plan, pos_and_or_path = [], []
        position_to_go = list(position_list)
        curr_pos, curr_or = start_motion_state

        # Get agent to goal position
        while position_to_go and curr_pos != goal_position:
            next_pos = position_to_go.pop(0)
            action = Action.determine_action_for_change_in_pos(
                curr_pos, next_pos)
            action_plan.append(action)
            curr_or = action if action != Action.STAY else curr_or
            pos_and_or_path.append((next_pos, curr_or))
            curr_pos = next_pos

        # Fix agent orientation if necessary
        if curr_or != goal_orientation:
            new_pos, _ = self.mdp._move_if_direction(curr_pos, curr_or,
                                                     goal_orientation)
            # assert new_pos == goal_position
            action_plan.append(goal_orientation)
            pos_and_or_path.append((goal_position, goal_orientation))

        # Add interact action
        action_plan.append(Action.INTERACT)
        pos_and_or_path.append((goal_position, goal_orientation))

        return action_plan, pos_and_or_path, len(action_plan)

    def _graph_from_grid(self):
        """Creates a graph adjacency matrix from an Overcooked MDP class."""
        state_decoder = {}
        for state_index, motion_state in enumerate(
                self.mdp.get_valid_player_positions_and_orientations()):
            state_decoder[state_index] = motion_state

        pos_encoder = {
            motion_state: state_index
            for state_index, motion_state in state_decoder.items()
        }
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for state_index, start_motion_state in state_decoder.items():
            for (
                    action,
                    successor_motion_state,
            ) in self._get_valid_successor_motion_states(start_motion_state):
                adj_pos_index = pos_encoder[successor_motion_state]
                adjacency_matrix[state_index][
                    adj_pos_index] = self._graph_action_cost(action)

        return Graph(adjacency_matrix, pos_encoder, state_decoder)

    def _graph_action_cost(self, action):
        """Returns cost of a single-agent action"""
        assert action in Action.ALL_ACTIONS
        return 1

    def _get_valid_successor_motion_states(self, start_motion_state):
        """Get valid motion states one action away from the starting motion
        state."""
        start_position, start_orientation = start_motion_state
        return [(
            action,
            self.mdp._move_if_direction(start_position, start_orientation,
                                        action),
        ) for action in Action.ALL_ACTIONS]

    def min_cost_between_features(self,
                                  pos_list1,
                                  pos_list2,
                                  manhattan_if_fail=False):
        """
        Determines the minimum number of timesteps necessary for a player to go
        from any terrain feature in list1 to any feature in list2 and perform an
        interact action
        """
        min_dist = np.Inf
        min_manhattan = np.Inf
        for pos1, pos2 in itertools.product(pos_list1, pos_list2):
            for mg1, mg2 in itertools.product(self.motion_goals_for_pos[pos1],
                                              self.motion_goals_for_pos[pos2]):
                if not self.is_valid_motion_start_goal_pair(mg1, mg2):
                    if manhattan_if_fail:
                        pos0, pos1 = mg1[0], mg2[0]
                        curr_man_dist = manhattan_distance(pos0, pos1)
                        if curr_man_dist < min_manhattan:
                            min_manhattan = curr_man_dist
                    continue
                curr_dist = self.get_gridworld_distance(mg1, mg2)
                if curr_dist < min_dist:
                    min_dist = curr_dist

        # +1 to account for interaction action
        if manhattan_if_fail and min_dist == np.Inf:
            min_dist = min_manhattan
        min_cost = min_dist + 1
        return min_cost

    def min_cost_to_feature(self,
                            start_pos_and_or,
                            feature_pos_list,
                            with_argmin=False,
                            with_motion_goal=False,
                            debug=False):
        """
        Determines the minimum number of timesteps necessary for a player to go
        from the starting position and orientation to any feature in
        feature_pos_list and perform an interact action
        """
        start_pos = start_pos_and_or[0]
        assert self.mdp.get_terrain_type_at_pos(start_pos) != "X"
        min_dist = np.Inf
        best_feature = None
        best_motion_goal = None
        for feature_pos in feature_pos_list:
            for feature_goal in self.motion_goals_for_pos[feature_pos]:
                if not self.is_valid_motion_start_goal_pair(
                        start_pos_and_or, feature_goal, debug=debug):
                    continue
                curr_dist = self.get_gridworld_distance(start_pos_and_or,
                                                        feature_goal)
                if curr_dist < min_dist:
                    best_feature = feature_pos
                    best_motion_goal = feature_goal
                    min_dist = curr_dist
        # +1 to account for interaction action
        min_cost = min_dist + 1
        if with_motion_goal:
            return min_cost, best_motion_goal
        if with_argmin:
            # assert (best_feature
            #         is not None), f"{start_pos_and_or} vs {feature_pos_list}"
            return min_cost, best_feature
        return min_cost

    def _get_goal_dict(self):
        """Creates a dictionary of all possible goal states for all possible
        terrain features that the agent might want to interact with."""
        terrain_feature_locations = []
        for terrain_type, pos_list in self.mdp.terrain_pos_dict.items():
            if terrain_type != " ":
                terrain_feature_locations += pos_list
        return {
            feature_pos:
            self._get_possible_motion_goals_for_feature(feature_pos)
            for feature_pos in terrain_feature_locations
        }

    def _get_possible_motion_goals_for_feature(self, goal_pos):
        """Returns a list of possible goal positions (and orientations)
        that could be used for motion planning to get to goal_pos"""
        goals = []
        valid_positions = self.mdp.get_valid_player_positions()
        for d in Direction.ALL_DIRECTIONS:
            adjacent_pos = Action.move_in_direction(goal_pos, d)
            if adjacent_pos in valid_positions:
                goal_orientation = Direction.OPPOSITE_DIRECTIONS[d]
                motion_goal = (adjacent_pos, goal_orientation)
                goals.append(motion_goal)
        return goals


class JointMotionPlanner:
    """A planner that computes optimal plans for two agents to
    arrive at goal positions and orientations in a OvercookedGridworld.
    Args:
        mdp (OvercookedGridworld): gridworld of interest
    """

    def __init__(self, mdp, params, debug=False):
        self.mdp = mdp

        # Whether starting orientations should be accounted for
        # when solving all motion problems
        # (increases number of plans by a factor of 4)
        # but removes additional fudge factor <= 1 for each
        # joint motion plan
        self.start_orientations = params["start_orientations"]

        # Enable both agents to have the same motion goal
        self.same_motion_goals = params["same_motion_goals"]

        # Single agent motion planner
        self.motion_planner = MotionPlanner(
            mdp, counter_goals=params["counter_goals"])

        # Graph problem that returns optimal paths from
        # starting positions to goal positions (without
        # accounting for orientations)
        self.joint_graph_problem = self._joint_graph_from_grid()

    def get_low_level_action_plan(self,
                                  start_jm_state,
                                  goal_jm_state,
                                  merge_one=False):
        """
        Returns pre-computed plan from initial joint motion state
        to a goal joint motion state.

        Args:
            start_jm_state (tuple): starting pos & orients
                ((pos1, or1), (pos2, or2))
            goal_jm_state (tuple): goal pos & orients ((pos1, or1), (pos2, or2))

        Returns:
            joint_action_plan (list): joint actions to be executed to reach
                end_jm_state
            end_jm_state (tuple): the pair of (pos, or) tuples corresponding
                to the ending timestep (this will usually be different from
                goal_jm_state, as one agent will end before other).
            plan_lengths (tuple): lengths for each agent's plan
        """
        assert self.is_valid_joint_motion_pair(
            start_jm_state, goal_jm_state
        ), f"start: {start_jm_state} \t end: {goal_jm_state} was not a valid " \
           f"motion goal pair"

        if self.start_orientations:
            plan_key = (start_jm_state, goal_jm_state)
        else:
            starting_positions = tuple(
                player_pos_and_or[0] for player_pos_and_or in start_jm_state)
            goal_positions = tuple(
                player_pos_and_or[0] for player_pos_and_or in goal_jm_state)
            # If beginning positions are equal to end positions, the pre-stored
            # plan (not dependent on initial positions) will likely return a
            # wrong answer, so we compute it from scratch.
            #
            # This is because we only compute plans with starting orientations
            # (North, North), so if one of the two agents starts at location X
            # with orientation East it's goal is to get to location X with
            # orientation North. The precomputed plan will just tell that agent
            # that it is already at the goal, so no actions (or just 'interact')
            # are necessary.
            #
            # We also compute the plan for any shared motion goal with SAFE_RUN,
            # as there are some minor edge cases that could not be accounted for
            # but I expect should not make a difference in nearly all scenarios
            if any([s == g for s, g in zip(starting_positions, goal_positions)
                   ]) or (SAFE_RUN and goal_positions[0] == goal_positions[1]):
                return self._obtain_plan(start_jm_state, goal_jm_state)

            dummy_orientation = Direction.NORTH
            dummy_start_jm_state = tuple(
                (pos, dummy_orientation) for pos in starting_positions)
            plan_key = (dummy_start_jm_state, goal_jm_state)

        joint_action_plan, end_jm_state, plan_lengths = self._obtain_plan(
            plan_key[0], plan_key[1], merge_one=merge_one)
        return joint_action_plan, end_jm_state, plan_lengths

    def is_valid_jm_start_goal_pair(self, joint_start_state, joint_goal_state):
        """Checks if the combination of joint start state and joint goal state
        is valid"""
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        check_valid_fn = self.motion_planner.is_valid_motion_start_goal_pair
        return all([
            check_valid_fn(joint_start_state[i], joint_goal_state[i])
            for i in range(2)
        ])

    def _obtain_plan(self,
                     joint_start_state,
                     joint_goal_state,
                     merge_one=False):
        """Either use motion planner or actually compute a joint plan"""
        # Try using MotionPlanner plans and join them together
        (
            action_plans,
            pos_and_or_paths,
            plan_lengths,
        ) = self._get_plans_from_single_planner(joint_start_state,
                                                joint_goal_state)
        # Check if individual plans conflict
        have_conflict = self.plans_have_conflict(joint_start_state,
                                                 joint_goal_state,
                                                 pos_and_or_paths, plan_lengths)
        # If there is no conflict, the joint plan computed by joining single
        # agent MotionPlanner plans is optimal
        if not have_conflict:
            (
                joint_action_plan,
                end_pos_and_orientations,
            ) = self._join_single_agent_action_plans(
                joint_start_state,
                action_plans,
                pos_and_or_paths,
                min(plan_lengths),
            )
            return joint_action_plan, end_pos_and_orientations, plan_lengths

        # If there is a conflict in the single motion plan and the agents have
        # the same goal, the graph problem can't be used either as it can't
        # handle same goal state: we compute manually what the best way to
        # handle the conflict is
        elif self._agents_are_in_same_position(joint_goal_state):
            (
                joint_action_plan,
                end_pos_and_orientations,
                plan_lengths,
            ) = self._handle_path_conflict_with_same_goal(
                joint_start_state,
                joint_goal_state,
                action_plans,
                pos_and_or_paths,
            )
            return joint_action_plan, end_pos_and_orientations, plan_lengths

        # If there is a conflict, and the agents have different goals, we can
        # use solve the joint graph problem return
        # self._compute_plan_from_joint_graph(joint_start_state,
        # joint_goal_state)

        try:
            return self._get_joint_plan_from_merging_ind_paths(
                pos_and_or_paths,
                joint_start_state,
                joint_goal_state,
                merge_one=merge_one,
            )
        except ValueError:
            return self._compute_plan_from_joint_graph(joint_start_state,
                                                       joint_goal_state)

    def merge_paths_dp(self, pos_and_or_paths, joint_start_state):
        """
        DP solver that merges two paths such that they do not have conflicts.
        Note that this solver can only deal with paths that does not share
        the same start point and end point.
        Args:
            pos_and_or_paths (list): list of tuple(position, orientation)
        Returns:
            position_list1 (list), position_list2 (list)
        """

        s1, s2 = self.extract_ind_pos_list(pos_and_or_paths, joint_start_state)

        if s1[-1] == s2[-1] or s1[0] == s2[0]:
            return None, None
        oo = np.inf
        table = np.full((len(s1) + 1, len(s2) + 1), oo)
        table[0, 0] = 0
        choice = np.full((len(s1) + 1, len(s2) + 1), -1)
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    table[i][j] = oo
                    continue
                ncost = table[i, j] + (1 if j >= i else 0)
                if ncost < table[i, j + 1]:
                    table[i, j + 1] = ncost
                    choice[i, j + 1] = 0
                ncost = table[i, j] + (1 if i >= j else 0)
                if ncost < table[i + 1, j]:
                    table[i + 1, j] = ncost
                    choice[i + 1, j] = 1
                ncost = table[i, j]
                if ncost < table[i + 1, j + 1]:
                    table[i + 1, j + 1] = ncost
                    choice[i + 1, j + 1] = 2
        # Use the choice matrix to build back the path
        i = len(s1) - 1
        j = len(s2) - 1
        ans1 = []
        ans2 = []
        while 0 < i or 0 < j:
            ans1.append(s1[i])
            ans2.append(s2[j])
            if choice[i, j] == 0:
                j -= 1
            elif choice[i, j] == 1:
                i -= 1
            elif choice[i, j] == 2:
                i -= 1
                j -= 1
            else:
                raise ValueError("Static agent blocking the way: No solution!")
        ans1.append(s1[0])
        ans2.append(s2[0])
        ans1.reverse()
        ans2.reverse()

        # paths are invalid if they crash into each other
        for idx in range(min(len(ans1), len(ans2)) - 1):
            if ans1[idx] == ans2[idx + 1] and ans1[idx + 1] == ans2[idx]:
                raise ValueError("Two paths crached: Solution not valid!")

        return ans1[1:], ans2[1:]

    def merge_one_path_into_other_dp(self, pos_and_or_paths, joint_start_state):
        """
        DP solver that merges one path to another by only changing one
        path's pos and or such that they do not have conflicts.
        Note that this solver can only deal with paths that does not share
        the same start point and end point.
        Args:
            pos_and_or_paths (list): list of tuple(position, orientation)
        Returns:
            position_list1 (list), position_list2 (list)
        """

        s1, s2 = self.extract_ind_pos_list(pos_and_or_paths, joint_start_state)

        if s1[-1] == s2[-1] or s1[0] == s2[0]:
            return None, None
        oo = np.inf
        table = np.full((len(s1) + 1, len(s2) + 1), oo)
        table[0, 0] = 0
        choice = np.full((len(s1) + 1, len(s2) + 1), -1)
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    table[i][j] = oo
                    continue
                ncost = table[i, j] + (1 if j >= i else 0)
                if ncost < table[i, j + 1]:
                    table[i, j + 1] = ncost
                    choice[i, j + 1] = 0
                ncost = table[i, j] + (1 if i >= j else 0)
                if ncost < table[i + 1, j]:
                    table[i + 1, j] = ncost
                    choice[i + 1, j] = 1
                ncost = table[i, j]
                if ncost < table[i + 1, j + 1]:
                    table[i + 1, j + 1] = ncost
                    choice[i + 1, j + 1] = 2
        # Use the choice matrix to build back the path
        i = len(s1) - 1
        j = len(s2) - 1
        ans1 = []
        ans2 = []
        while 0 < i or 0 < j:
            ans1.append(s1[i])
            ans2.append(s2[j])
            if choice[i, j] == 0:
                j -= 1
            elif choice[i, j] == 1:
                i -= 1
            elif choice[i, j] == 2:
                i -= 1
                j -= 1
            else:
                raise ValueError("Static agent blocking the way: No solution!")
        ans1.append(s1[0])
        ans2.append(s2[0])
        ans1.reverse()
        ans2.reverse()

        # paths are invalid if they crash into each other
        for idx in range(min(len(ans1), len(ans2)) - 1):
            if ans1[idx] == ans2[idx + 1] and ans1[idx + 1] == ans2[idx]:
                raise ValueError("Two paths crached: Solution not valid!")

        return ans1[1:], ans2[1:]

    def extract_ind_pos_list(self, pos_and_or_paths, joint_start_state):
        pos_and_or_path1, pos_and_or_path2 = pos_and_or_paths
        pos_list1 = [row[0] for row in pos_and_or_path1]
        pos_list2 = [row[0] for row in pos_and_or_path2]
        start1, start2 = joint_start_state
        pos_list1.insert(0, start1[0])
        pos_list2.insert(0, start2[0])
        return pos_list1, pos_list2

    def _get_joint_plan_from_merging_ind_paths(
        self,
        pos_and_or_paths,
        joint_start_state,
        joint_goal_state,
        merge_one=False,
    ):
        """
        Get joint motion plan by using the DP solver to resolve conflicts
        in the individual motion paths
        Args:
            pos_and_or_paths (list): list of (pos, or) pairs visited during
                                    plan execution
                                    (not including start, but including goal)
            joint_start_state (list(tuple)): list of starting position and
                                             orientation
            joint_goal_state (list(tuple)): list of goal position and
                                            orientation
        """
        # resolve conflict in the individual paths
        if merge_one:
            path_lists = self.merge_one_path_into_other_dp(
                pos_and_or_paths, joint_start_state)
        else:
            path_lists = self.merge_paths_dp(pos_and_or_paths,
                                             joint_start_state)

        # obtain action_plans from paths
        action_plans, pos_and_or_paths, plan_lengths = [], [], []
        for path_list, start, goal in zip(path_lists, joint_start_state,
                                          joint_goal_state):
            (
                action_plan,
                pos_and_or_path,
                plan_length,
            ) = self.motion_planner.action_plan_from_positions(
                path_list, start, goal)
            action_plans.append(action_plan)
            pos_and_or_paths.append(pos_and_or_path)
            plan_lengths.append(plan_length)

        # joint the action plans
        (
            joint_action_plan,
            end_pos_and_orientations,
        ) = self._join_single_agent_action_plans(joint_start_state,
                                                 action_plans, pos_and_or_paths,
                                                 min(plan_lengths))
        return joint_action_plan, end_pos_and_orientations, plan_lengths

    def _get_plans_from_single_planner(self, joint_start_state,
                                       joint_goal_state):
        """
        Get individual action plans for each agent from the MotionPlanner to get
        each agent independently to their goal state. NOTE: these plans might
        conflict
        """
        single_agent_motion_plans = [
            self.motion_planner.get_plan(start, goal)
            for start, goal in zip(joint_start_state, joint_goal_state)
        ]
        action_plans, pos_and_or_paths = [], []
        for action_plan, pos_and_or_path, _ in single_agent_motion_plans:
            action_plans.append(action_plan)
            pos_and_or_paths.append(pos_and_or_path)
        plan_lengths = tuple(len(p) for p in action_plans)
        assert all(
            [plan_lengths[i] == len(pos_and_or_paths[i]) for i in range(2)])
        return action_plans, pos_and_or_paths, plan_lengths

    def plans_have_conflict(
        self,
        joint_start_state,
        joint_goal_state,
        pos_and_or_paths,
        plan_lengths,
    ):
        """Check if the sequence of pos_and_or_paths for the two agents
        conflict"""
        min_length = min(plan_lengths)
        prev_positions = tuple(s[0] for s in joint_start_state)
        for t in range(min_length):
            curr_pos_or0, curr_pos_or1 = (
                pos_and_or_paths[0][t],
                pos_and_or_paths[1][t],
            )
            curr_positions = (curr_pos_or0[0], curr_pos_or1[0])
            if self.mdp.is_transition_collision(prev_positions, curr_positions):
                return True
            prev_positions = curr_positions
        return False

    def _join_single_agent_action_plans(self, joint_start_state, action_plans,
                                        pos_and_or_paths, finishing_time):
        """Returns the joint action plan and end joint state obtained by joining
        the individual action plans"""
        assert finishing_time > 0
        end_joint_state = (
            pos_and_or_paths[0][finishing_time - 1],
            pos_and_or_paths[1][finishing_time - 1],
        )
        joint_action_plan = list(
            zip(*[
                action_plans[0][:finishing_time],
                action_plans[1][:finishing_time],
            ]))
        return joint_action_plan, end_joint_state

    def _handle_path_conflict_with_same_goal(
        self,
        joint_start_state,
        joint_goal_state,
        action_plans,
        pos_and_or_paths,
    ):
        """Assumes that optimal path in case two agents have the same goal and
        their paths conflict is for one of the agents to wait. Checks resulting
        plans if either agent waits, and selects the shortest cost among the
        two."""

        (
            joint_plan0,
            end_pos_and_or0,
            plan_lengths0,
        ) = self._handle_conflict_with_same_goal_idx(
            joint_start_state,
            joint_goal_state,
            action_plans,
            pos_and_or_paths,
            wait_agent_idx=0,
        )

        (
            joint_plan1,
            end_pos_and_or1,
            plan_lengths1,
        ) = self._handle_conflict_with_same_goal_idx(
            joint_start_state,
            joint_goal_state,
            action_plans,
            pos_and_or_paths,
            wait_agent_idx=1,
        )

        assert any([joint_plan0 is not None, joint_plan1 is not None])

        best_plan_idx = np.argmin([min(plan_lengths0), min(plan_lengths1)])
        solutions = [
            (joint_plan0, end_pos_and_or0, plan_lengths0),
            (joint_plan1, end_pos_and_or1, plan_lengths1),
        ]
        return solutions[best_plan_idx]

    def _handle_conflict_with_same_goal_idx(
        self,
        joint_start_state,
        joint_goal_state,
        action_plans,
        pos_and_or_paths,
        wait_agent_idx,
    ):
        """
        Determines what is the best joint plan if whenether there is a conflict
        between the two agents' actions, the agent with index `wait_agent_idx`
        waits one turn.

        If the agent that is assigned to wait is "in front" of the non-waiting
        agent, this could result in an endless conflict. In this case, we return
        infinite finishing times.
        """
        idx0, idx1 = 0, 0
        prev_positions = [
            start_pos_and_or[0] for start_pos_and_or in joint_start_state
        ]
        curr_pos_or0, curr_pos_or1 = joint_start_state

        agent0_plan_original, agent1_plan_original = action_plans

        joint_plan = []
        # While either agent hasn't finished their plan
        while idx0 != len(agent0_plan_original) and idx1 != len(
                agent1_plan_original):
            next_pos_or0, next_pos_or1 = (
                pos_and_or_paths[0][idx0],
                pos_and_or_paths[1][idx1],
            )
            next_positions = (next_pos_or0[0], next_pos_or1[0])

            # If agents collide, let the waiting agent wait and the non-waiting
            # agent take a step
            if self.mdp.is_transition_collision(prev_positions, next_positions):
                if wait_agent_idx == 0:
                    curr_pos_or0 = (
                        curr_pos_or0  # Agent 0 will wait, stays the same
                    )
                    curr_pos_or1 = next_pos_or1
                    curr_joint_action = [
                        Action.STAY,
                        agent1_plan_original[idx1],
                    ]
                    idx1 += 1
                elif wait_agent_idx == 1:
                    curr_pos_or0 = next_pos_or0
                    curr_pos_or1 = (
                        curr_pos_or1  # Agent 1 will wait, stays the same
                    )
                    curr_joint_action = [
                        agent0_plan_original[idx0],
                        Action.STAY,
                    ]
                    idx0 += 1

                curr_positions = (curr_pos_or0[0], curr_pos_or1[0])

                # If one agent waiting causes other to crash into it, return
                # None
                if self._agents_are_in_same_position(
                    (curr_pos_or0, curr_pos_or1)):
                    return None, None, [np.Inf, np.Inf]

            else:
                curr_pos_or0, curr_pos_or1 = next_pos_or0, next_pos_or1
                curr_positions = next_positions
                curr_joint_action = [
                    agent0_plan_original[idx0],
                    agent1_plan_original[idx1],
                ]
                idx0 += 1
                idx1 += 1

            joint_plan.append(curr_joint_action)
            prev_positions = curr_positions

        assert idx0 != idx1, "No conflict found"

        end_pos_and_or = (curr_pos_or0, curr_pos_or1)
        finishing_times = ((np.Inf, idx1) if wait_agent_idx == 0 else
                           (idx0, np.Inf))
        return joint_plan, end_pos_and_or, finishing_times

    def is_valid_joint_motion_goal(self, joint_goal_state):
        """Checks whether the goal joint positions and orientations are a valid
        goal"""
        if not self.same_motion_goals and self._agents_are_in_same_position(
                joint_goal_state):
            return False
        multi_cc_map = (len(
            self.motion_planner.graph_problem.connected_components) > 1)
        players_in_same_cc = self.motion_planner.graph_problem.are_in_same_cc(
            joint_goal_state[0], joint_goal_state[1])
        if multi_cc_map and players_in_same_cc:
            return False
        return all([
            self.motion_planner.is_valid_motion_goal(player_state)
            for player_state in joint_goal_state
        ])

    def is_valid_joint_motion_pair(self, joint_start_state, joint_goal_state):
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        return all([
            self.motion_planner.is_valid_motion_start_goal_pair(
                joint_start_state[i], joint_goal_state[i]) for i in range(2)
        ])

    def _agents_are_in_same_position(self, joint_motion_state):
        agent_positions = [
            player_pos_and_or[0] for player_pos_and_or in joint_motion_state
        ]
        return len(agent_positions) != len(set(agent_positions))

    def _compute_plan_from_joint_graph(self, joint_start_state,
                                       joint_goal_state):
        """Compute joint action plan for two agents to achieve a
        certain position and orientation with the joint motion graph

        Args:
            joint_start_state: pair of start (pos, or)
            goal_statuses: pair of goal (pos, or)
        """
        assert self.is_valid_joint_motion_pair(
            joint_start_state, joint_goal_state), joint_goal_state
        # Solve shortest-path graph problem
        start_positions = list(zip(*joint_start_state))[0]
        goal_positions = list(zip(*joint_goal_state))[0]
        joint_positions_node_path = self.joint_graph_problem.get_node_path(
            start_positions, goal_positions)[1:]
        (
            joint_actions_list,
            end_pos_and_orientations,
            finishing_times,
        ) = self.joint_action_plan_from_positions(joint_positions_node_path,
                                                  joint_start_state,
                                                  joint_goal_state)
        return joint_actions_list, end_pos_and_orientations, finishing_times

    def joint_action_plan_from_positions(self, joint_positions,
                                         joint_start_state, joint_goal_state):
        """
        Finds an action plan and it's cost, such that at least one of the agent
        goal states is achieved
        Args:
            joint_positions (list): list of joint positions to be reached after
                the starting position (does not include starting position, but
                includes ending position)
            joint_start_state (tuple): pair of starting positions and
                orientations
            joint_goal_state (tuple): pair of goal positions and orientations
        """
        action_plans = []
        for i in range(2):
            agent_position_sequence = [
                joint_position[i] for joint_position in joint_positions
            ]
            action_plan, _, _ = self.motion_planner.action_plan_from_positions(
                agent_position_sequence,
                joint_start_state[i],
                joint_goal_state[i],
            )
            action_plans.append(action_plan)

        finishing_times = tuple(len(plan) for plan in action_plans)
        trimmed_action_plans = self._fix_plan_lengths(action_plans)
        joint_action_plan = list(zip(*trimmed_action_plans))
        end_pos_and_orientations = self._rollout_end_pos_and_or(
            joint_start_state, joint_action_plan)
        return joint_action_plan, end_pos_and_orientations, finishing_times

    def _fix_plan_lengths(self, plans):
        """Truncates the longer plan when shorter plan ends"""
        plans = list(plans)
        finishing_times = [len(p) for p in plans]
        delta_length = max(finishing_times) - min(finishing_times)
        if delta_length != 0:
            index_long_plan = np.argmax(finishing_times)
            long_plan = plans[index_long_plan]
            long_plan = long_plan[:min(finishing_times)]
        return plans

    def _rollout_end_pos_and_or(self, joint_start_state, joint_action_plan):
        """Execute plan in environment to determine ending positions and
        orientations"""
        # Assumes that final pos and orientations only depend on initial ones
        # (not on objects and other aspects of state).
        # Also assumes can't deliver more than two orders in one motion goal
        # (otherwise Environment will terminate)
        dummy_state = OvercookedState.from_players_pos_and_or(
            joint_start_state, order_list=["any", "any"])
        env = OvercookedEnv.from_mdp(
            self.mdp, horizon=200
        )  # Plans should be shorter than 200 timesteps, or something is likely wrong
        successor_state, is_done = env.execute_plan(dummy_state,
                                                    joint_action_plan)
        assert not is_done
        return successor_state.players_pos_and_or

    def _joint_graph_from_grid(self):
        """Creates a graph instance from the mdp instance. Each graph node
        encodes a pair of positions"""
        state_decoder = {}
        # Valid positions pairs, not including ones with both players in same
        # spot
        valid_joint_positions = self.mdp.get_valid_joint_player_positions()
        for state_index, joint_pos in enumerate(valid_joint_positions):
            state_decoder[state_index] = joint_pos

        state_encoder = {v: k for k, v in state_decoder.items()}
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for start_state_index, start_joint_positions in state_decoder.items():
            for (
                    joint_action,
                    successor_jm_state,
            ) in self._get_valid_successor_joint_positions(
                    start_joint_positions).items():
                successor_node_index = state_encoder[successor_jm_state]

                this_action_cost = self._graph_joint_action_cost(joint_action)
                current_cost = adjacency_matrix[start_state_index][
                    successor_node_index]

                if current_cost == 0 or this_action_cost < current_cost:
                    adjacency_matrix[start_state_index][
                        successor_node_index] = this_action_cost

        return Graph(adjacency_matrix, state_encoder, state_decoder)

    def _graph_joint_action_cost(self, joint_action):
        """The cost used in the graph shortest-path problem for a certain
        joint-action"""
        num_of_non_stay_actions = len(
            [a for a in joint_action if a != Action.STAY])
        # NOTE: Removing the possibility of having 0 cost joint_actions
        if num_of_non_stay_actions == 0:
            return 1
        return num_of_non_stay_actions

    def _get_valid_successor_joint_positions(self, starting_positions):
        """Get all joint positions that can be reached by a joint action.
        NOTE: this DOES NOT include joint positions with superimposed agents."""
        successor_joint_positions = {}
        joint_motion_actions = itertools.product(Action.MOTION_ACTIONS,
                                                 Action.MOTION_ACTIONS)

        # Under assumption that orientation doesn't matter
        dummy_orientation = Direction.NORTH
        dummy_player_states = [
            PlayerState(pos, dummy_orientation) for pos in starting_positions
        ]
        for joint_action in joint_motion_actions:
            new_positions, _ = self.mdp.compute_new_positions_and_orientations(
                dummy_player_states, joint_action)
            successor_joint_positions[joint_action] = new_positions
        return successor_joint_positions

    def derive_state(self, start_state, end_pos_and_ors, action_plans):
        """
        Given a start state, end position and orientations, and an action plan,
        recovers the resulting state without executing the entire plan.
        """
        if len(action_plans) == 0:
            return start_state

        end_state = start_state.deepcopy()
        end_players = []
        for player, end_pos_and_or in zip(end_state.players, end_pos_and_ors):
            new_player = player.deepcopy()
            position, orientation = end_pos_and_or
            new_player.update_pos_and_or(position, orientation)
            end_players.append(new_player)

        end_state.players = tuple(end_players)

        # Resolve environment effects for t - 1 turns
        plan_length = len(action_plans)
        assert plan_length > 0
        for _ in range(plan_length - 1):
            self.mdp.step_environment_effects(end_state)

        # Interacts
        last_joint_action = tuple(a if a == Action.INTERACT else Action.STAY
                                  for a in action_plans[-1])

        events_dict = {
            k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES
        }
        self.mdp.resolve_interacts(end_state, last_joint_action, events_dict)
        self.mdp.resolve_movement(end_state, last_joint_action)
        self.mdp.step_environment_effects(end_state)
        return end_state


class MediumLevelActionManager:
    """
    Manager for medium level actions (specific joint motion goals).
    Determines available medium level actions for each state.

    Args:
        mdp (OvercookedGridWorld): gridworld of interest
    """

    def __init__(self, mdp, params):
        # start_time = time.time()
        self.mdp = mdp

        self.params = params
        self.wait_allowed = params["wait_allowed"]
        self.counter_drop = params["counter_drop"]
        self.counter_pickup = params["counter_pickup"]

        self.joint_motion_planner = JointMotionPlanner(mdp, params)
        self.motion_planner = self.joint_motion_planner.motion_planner
        # logger.info(f"It took {time.time() - start_time} seconds to create "
        #             f"MediumLevelActionManager")

    def save_to_file(self, filename):
        with open(filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def joint_ml_actions(self, state):
        """Determine all possible joint medium level actions for a certain
        state"""
        agent1_actions, agent2_actions = tuple(
            self.get_medium_level_actions(state, player)
            for player in state.players)
        joint_ml_actions = list(
            itertools.product(agent1_actions, agent2_actions))

        # ml actions are nothing but specific joint motion goals
        valid_joint_ml_actions = list(
            filter(lambda a: self.is_valid_ml_action(state, a),
                   joint_ml_actions))

        # HACK: Could cause things to break.
        # Necessary to prevent states without successors (due to no counters
        # being allowed and no wait actions) causing A* to not find a solution
        if len(valid_joint_ml_actions) == 0:
            agent1_actions, agent2_actions = tuple(
                self.get_medium_level_actions(
                    state, player, waiting_substitute=True)
                for player in state.players)
            joint_ml_actions = list(
                itertools.product(agent1_actions, agent2_actions))
            valid_joint_ml_actions = list(
                filter(
                    lambda a: self.is_valid_ml_action(state, a),
                    joint_ml_actions,
                ))
            if len(valid_joint_ml_actions) == 0:
                logger.info(
                    f"WARNING: Found state without valid actions even after "
                    f"adding waiting substitute actions. State: {state}")
        return valid_joint_ml_actions

    def is_valid_ml_action(self, state, ml_action):
        return self.joint_motion_planner.is_valid_jm_start_goal_pair(
            state.players_pos_and_or, ml_action)

    def get_medium_level_actions(self, state, player, waiting_substitute=False):
        """
        Determine valid medium level actions for a player.

        Args:
            state (OvercookedState): current state
            waiting_substitute (bool): add a substitute action that takes the
                place of a waiting action (going to closest feature)

        Returns:
            player_actions (list): possible motion goals (pairs of goal
                positions and orientations)
        """
        player_actions = []
        counter_pickup_objects = self.mdp.get_counter_objects_dict(
            state, self.counter_pickup)
        if not player.has_object():
            onion_pickup = self.pickup_onion_actions(counter_pickup_objects)
            tomato_pickup = self.pickup_tomato_actions(counter_pickup_objects)
            dish_pickup = self.pickup_dish_actions(counter_pickup_objects)
            soup_pickup = self.pickup_counter_soup_actions(
                counter_pickup_objects)
            player_actions.extend(onion_pickup + tomato_pickup + dish_pickup +
                                  soup_pickup)

        else:
            player_object = player.get_object()
            pot_states_dict = self.mdp.get_pot_states(state)

            # No matter the object, we can place it on a counter
            if len(self.counter_drop) > 0:
                player_actions.extend(self.place_obj_on_counter_actions(state))

            if player_object.name == "soup":
                player_actions.extend(self.deliver_soup_actions())
            elif player_object.name == "onion":
                player_actions.extend(
                    self.put_onion_in_pot_actions(pot_states_dict))
            elif player_object.name == "tomato":
                player_actions.extend(
                    self.put_tomato_in_pot_actions(pot_states_dict))
            elif player_object.name == "dish":
                # Not considering all pots (only ones close to ready) to reduce
                # computation
                # NOTE: could try to calculate which pots are eligible, but
                # would probably take a lot of compute
                player_actions.extend(
                    self.pickup_soup_with_dish_actions(pot_states_dict,
                                                       only_nearly_ready=False))
            else:
                raise ValueError("Unrecognized object")

        if self.wait_allowed:
            player_actions.extend(self.wait_actions(player))

        if waiting_substitute:
            # Trying to mimic a "WAIT" action by adding the closest allowed
            # feature to the avaliable actions
            # This is because motion plans that aren't facing terrain features
            # (non counter, non empty spots) are not considered valid
            player_actions.extend(self.go_to_closest_feature_actions(player))

        is_valid_goal_given_start = lambda goal: self.motion_planner.is_valid_motion_start_goal_pair(
            player.pos_and_or, goal)
        player_actions = list(filter(is_valid_goal_given_start, player_actions))
        return player_actions

    def pickup_onion_actions(self, counter_objects, only_use_dispensers=False):
        """If only_use_dispensers is True, then only take onions from the
        dispensers"""
        onion_pickup_locations = self.mdp.get_onion_dispenser_locations()
        if not only_use_dispensers:
            onion_pickup_locations += counter_objects["onion"]
        return self._get_ml_actions_for_positions(onion_pickup_locations)

    def pickup_tomato_actions(self, counter_objects):
        tomato_dispenser_locations = self.mdp.get_tomato_dispenser_locations()
        tomato_pickup_locations = (tomato_dispenser_locations +
                                   counter_objects["tomato"])
        return self._get_ml_actions_for_positions(tomato_pickup_locations)

    def pickup_dish_actions(self, counter_objects, only_use_dispensers=False):
        """If only_use_dispensers is True, then only take dishes from the
        dispensers"""
        dish_pickup_locations = self.mdp.get_dish_dispenser_locations()
        if not only_use_dispensers:
            dish_pickup_locations += counter_objects["dish"]
        return self._get_ml_actions_for_positions(dish_pickup_locations)

    def pickup_counter_soup_actions(self, counter_objects):
        soup_pickup_locations = counter_objects["soup"]
        return self._get_ml_actions_for_positions(soup_pickup_locations)

    def place_obj_on_counter_actions(self, state):
        all_empty_counters = set(self.mdp.get_empty_counter_locations(state))
        valid_empty_counters = [
            c_pos for c_pos in self.counter_drop if c_pos in all_empty_counters
        ]
        return self._get_ml_actions_for_positions(valid_empty_counters)

    def deliver_soup_actions(self):
        serving_locations = self.mdp.get_serving_locations()
        return self._get_ml_actions_for_positions(serving_locations)

    def put_onion_in_pot_actions(self, pot_states_dict):
        partially_full_onion_pots = pot_states_dict["onion"]["partially_full"]
        fillable_pots = partially_full_onion_pots + pot_states_dict["empty"]
        return self._get_ml_actions_for_positions(fillable_pots)

    def put_tomato_in_pot_actions(self, pot_states_dict):
        partially_full_tomato_pots = pot_states_dict["tomato"]["partially_full"]
        fillable_pots = partially_full_tomato_pots + pot_states_dict["empty"]
        return self._get_ml_actions_for_positions(fillable_pots)

    def pickup_soup_with_dish_actions(self,
                                      pot_states_dict,
                                      only_nearly_ready=False):
        ready_pot_locations = (pot_states_dict["onion"]["ready"] +
                               pot_states_dict["tomato"]["ready"])
        nearly_ready_pot_locations = (pot_states_dict["onion"]["cooking"] +
                                      pot_states_dict["tomato"]["cooking"])
        if not only_nearly_ready:
            partially_full_pots = (pot_states_dict["tomato"]["partially_full"] +
                                   pot_states_dict["onion"]["partially_full"])
            nearly_ready_pot_locations = (nearly_ready_pot_locations +
                                          pot_states_dict["empty"] +
                                          partially_full_pots)
        return self._get_ml_actions_for_positions(ready_pot_locations +
                                                  nearly_ready_pot_locations)

    def go_to_closest_feature_actions(self, player):
        feature_locations = (self.mdp.get_onion_dispenser_locations() +
                             self.mdp.get_tomato_dispenser_locations() +
                             self.mdp.get_pot_locations() +
                             self.mdp.get_dish_dispenser_locations())
        closest_feature_pos = self.motion_planner.min_cost_to_feature(
            player.pos_and_or, feature_locations, with_argmin=True)[1]
        return self._get_ml_actions_for_positions([closest_feature_pos])

    def go_to_closest_feature_or_counter_to_goal(self, goal_pos_and_or,
                                                 goal_location):
        """Instead of going to goal_pos_and_or, go to the closest feature or
        counter to this goal, that ISN'T the goal itself"""
        valid_locations = (self.mdp.get_onion_dispenser_locations() +
                           self.mdp.get_tomato_dispenser_locations() +
                           self.mdp.get_pot_locations() +
                           self.mdp.get_dish_dispenser_locations() +
                           self.counter_drop)
        valid_locations.remove(goal_location)
        closest_non_goal_feature_pos = self.motion_planner.min_cost_to_feature(
            goal_pos_and_or, valid_locations, with_argmin=True)[1]
        return self._get_ml_actions_for_positions(
            [closest_non_goal_feature_pos])

    def wait_actions(self, player):
        waiting_motion_goal = (player.position, player.orientation)
        return [waiting_motion_goal]

    def _get_ml_actions_for_positions(self, positions_list):
        """Determine what are the ml actions (joint motion goals) for a list of
        positions

        Args:
            positions_list (list): list of target terrain feature positions
        """
        possible_motion_goals = []
        for pos in positions_list:
            # All possible ways to reach the target feature
            for (
                    motion_goal
            ) in self.joint_motion_planner.motion_planner.motion_goals_for_pos[
                    pos]:
                possible_motion_goals.append(motion_goal)
        return possible_motion_goals


class MediumLevelPlanner:
    """
    A planner that computes optimal plans for two agents to deliver a certain
    number of dishes in an OvercookedGridworld using medium level actions
    (single motion goals) in the corresponding A* search problem.
    """

    def __init__(self, mdp, mlp_params, ml_action_manager=None):
        self.mdp = mdp
        self.params = mlp_params
        self.ml_action_manager = (ml_action_manager if ml_action_manager else
                                  MediumLevelActionManager(mdp, mlp_params))
        self.jmp = self.ml_action_manager.joint_motion_planner
        self.mp = self.jmp.motion_planner

    @staticmethod
    def from_action_manager_file(filename):
        mlp_action_manager = load_saved_action_manager(filename)
        mdp = mlp_action_manager.mdp
        params = mlp_action_manager.params
        return MediumLevelPlanner(mdp, params, mlp_action_manager)

    @staticmethod
    def from_pickle_or_compute(mdp,
                               mlp_params,
                               custom_filename=None,
                               force_compute=False,
                               info=True):
        assert isinstance(mdp, OvercookedGridworld)

        filename = (custom_filename if custom_filename is not None else
                    mdp.layout_name + "_am.pkl")

        if force_compute:
            return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)

        try:
            mlp = MediumLevelPlanner.from_action_manager_file(filename)

            if mlp.ml_action_manager.params != mlp_params or mlp.mdp != mdp:
                logger.info(
                    "Mlp with different params or mdp found, computing from "
                    "scratch")
                return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)

        except (
                FileNotFoundError,
                ModuleNotFoundError,
                EOFError,
                AttributeError,
        ) as e:
            logger.info(f"Recomputing planner due to: {e}")
            return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)

        if info:
            logger.info(f"Loaded MediumLevelPlanner from "
                        f"{os.path.join(PLANNERS_DIR, filename)}")
        return mlp

    @staticmethod
    def compute_mlp(filename, mdp, mlp_params):
        final_filepath = os.path.join(PLANNERS_DIR, filename)
        logger.info(
            f"Computing MediumLevelPlanner to be saved in {final_filepath}")
        # start_time = time.time()
        mlp = MediumLevelPlanner(mdp, mlp_params=mlp_params)
        # logger.info(f"It took {time.time() -start_time} seconds to create mlp")
        mlp.ml_action_manager.save_to_file(final_filepath)
        return mlp

    def get_low_level_action_plan(
        self,
        start_state,
        h_fn,
        delivery_horizon=4,
        debug=False,
        goal_info=False,
    ):
        """
        Get a plan of joint-actions executable in the environment that will lead
        to a goal number of deliveries

        Args:
            state (OvercookedState): starting state

        Returns:
            full_joint_action_plan (list): joint actions to reach goal
        """
        start_state = start_state.deepcopy()
        ml_plan, cost = self.get_ml_plan(start_state,
                                         h_fn,
                                         delivery_horizon=delivery_horizon,
                                         debug=debug)

        if start_state.order_list is None:
            start_state.order_list = ["any"] * delivery_horizon

        full_joint_action_plan = self.get_low_level_plan_from_ml_plan(
            start_state, ml_plan, h_fn, debug=debug, goal_info=goal_info)
        assert cost == len(full_joint_action_plan), (
            f"A* cost {cost} but full joint action plan cost "
            f"{len(full_joint_action_plan)}")
        if debug:
            logger.info(f"Found plan with cost {cost}")
        return full_joint_action_plan

    def get_low_level_plan_from_ml_plan(self,
                                        start_state,
                                        ml_plan,
                                        heuristic_fn,
                                        debug=False,
                                        goal_info=False):
        t = 0
        full_joint_action_plan = []
        curr_state = start_state
        curr_motion_state = start_state.players_pos_and_or
        prev_h = heuristic_fn(start_state, t, debug=False)

        if len(ml_plan) > 0 and goal_info:
            logger.info(f"First motion goal: {ml_plan[0][0]}")

        if debug:
            logger.info("Start state")
            # TODO: print_state not defined
            OvercookedEnv.print_state(self.mdp, start_state)

        for joint_motion_goal, goal_state in ml_plan:
            (
                joint_action_plan,
                end_motion_state,
                plan_costs,
            ) = self.ml_action_manager.joint_motion_planner.get_low_level_action_plan(
                curr_motion_state, joint_motion_goal)
            curr_plan_cost = min(plan_costs)
            full_joint_action_plan.extend(joint_action_plan)
            t += 1

            if debug:
                logger.info(t)
                # TODO: print_state not defined
                OvercookedEnv.print_state(self.mdp, goal_state)

            if SAFE_RUN:
                env = OvercookedEnv.from_mdp(self.mdp,
                                             info_level=0,
                                             horizon=100)
                s_prime, _ = OvercookedEnv.execute_plan(env, curr_state,
                                                        joint_action_plan)
                assert s_prime == goal_state

            curr_h = heuristic_fn(goal_state, t, debug=False)
            self.check_heuristic_consistency(curr_h, prev_h, curr_plan_cost)
            curr_motion_state, prev_h, curr_state = (
                end_motion_state,
                curr_h,
                goal_state,
            )
        return full_joint_action_plan

    @staticmethod
    def check_heuristic_consistency(curr_heuristic_val, prev_heuristic_val,
                                    actual_edge_cost):
        delta_h = curr_heuristic_val - prev_heuristic_val
        assert actual_edge_cost >= delta_h, (
            f"Heuristic was not consistent. \n Prev h: {prev_heuristic_val}, "
            f"Curr h: {curr_heuristic_val}, Actual cost: {actual_edge_cost}, "
            f"Δh: {delta_h}")

    def get_ml_plan(self, start_state, h_fn, delivery_horizon=4, debug=False):
        """
        Solves A* Search problem to find optimal sequence of medium level actions
        to reach the goal number of deliveries

        Returns:
            ml_plan (list): plan not including starting state in form
                [(joint_action, successor_state), ..., (joint_action, goal_state)]
            cost (int): A* Search cost
        """
        start_state = start_state.deepcopy()
        if start_state.order_list is None:
            start_state.order_list = ["any"] * delivery_horizon
        else:
            start_state.order_list = start_state.order_list[:delivery_horizon]

        expand_fn = lambda state: self.get_successor_states(state)
        goal_fn = lambda state: state.num_orders_remaining == 0
        heuristic_fn = lambda state: h_fn(state)

        search_problem = SearchTree(start_state,
                                    goal_fn,
                                    expand_fn,
                                    heuristic_fn,
                                    max_iter_count=1e6,
                                    debug=debug)
        ml_plan, cost = search_problem.A_star_graph_search(info=True)
        return ml_plan[1:], cost

    def get_successor_states(self, start_state):
        """Successor states for medium-level actions are defined as
        the first state in the corresponding motion plan in which
        one of the two agents' subgoals is satisfied.

        Returns: list of
            joint_motion_goal: ((pos1, or1), (pos2, or2)) specifying the
                                motion plan goal for both agents

            successor_state:   OvercookedState corresponding to state
                               arrived at after executing part of the motion plan
                               (until one of the agents arrives at his goal status)

            plan_length:       Time passed until arrival to the successor state
        """
        if self.mdp.is_terminal(start_state):
            return []

        start_jm_state = start_state.players_pos_and_or
        successor_states = []
        for goal_jm_state in self.ml_action_manager.joint_ml_actions(
                start_state):
            (
                joint_motion_action_plans,
                end_pos_and_ors,
                plan_costs,
            ) = self.jmp.get_low_level_action_plan(start_jm_state,
                                                   goal_jm_state)
            end_state = self.jmp.derive_state(start_state, end_pos_and_ors,
                                              joint_motion_action_plans)

            if SAFE_RUN:
                assert (end_pos_and_ors[0] == goal_jm_state[0] or
                        end_pos_and_ors[1] == goal_jm_state[1])
                env = OvercookedEnv.from_mdp(self.mdp,
                                             info_level=0,
                                             horizon=100)
                s_prime, _ = OvercookedEnv.execute_plan(
                    env, start_state, joint_motion_action_plans, display=False)
                assert end_state == s_prime

            successor_states.append((goal_jm_state, end_state, min(plan_costs)))
        return successor_states

    def get_successor_states_fixed_other(self, start_state, other_agent,
                                         other_agent_idx):
        """
        Get the successor states of a given start state, assuming that the other
        agent is fixed and will act according to the passed in model
        """
        if self.mdp.is_terminal(start_state):
            return []

        player = start_state.players[1 - other_agent_idx]
        ml_actions = self.ml_action_manager.get_medium_level_actions(
            start_state, player)

        if len(ml_actions) == 0:
            ml_actions = self.ml_action_manager.get_medium_level_actions(
                start_state, player, waiting_substitute=True)

        successor_high_level_states = []
        for ml_action in ml_actions:
            (
                action_plan,
                end_state,
                cost,
            ) = self.get_embedded_low_level_action_plan(start_state, ml_action,
                                                        other_agent,
                                                        other_agent_idx)

            if not self.mdp.is_terminal(end_state):
                # Adding interact action and deriving last state
                other_agent_action, _ = other_agent.action(end_state)
                last_joint_action = ((Action.INTERACT, other_agent_action)
                                     if other_agent_idx == 1 else
                                     (other_agent_action, Action.INTERACT))
                action_plan = action_plan + (last_joint_action,)
                cost = cost + 1

                end_state, _ = self.embedded_mdp_step(
                    end_state,
                    Action.INTERACT,
                    other_agent_action,
                    other_agent.agent_index,
                )

            successor_high_level_states.append((action_plan, end_state, cost))
        return successor_high_level_states

    def get_embedded_low_level_action_plan(self, state, goal_pos_and_or,
                                           other_agent, other_agent_idx):
        """Find action plan for a specific motion goal with A* considering the
        other agent"""
        other_agent.set_agent_index(other_agent_idx)
        agent_idx = 1 - other_agent_idx

        expand_fn = lambda state: self.embedded_mdp_succ_fn(state, other_agent)
        goal_fn = (lambda state: state.players[agent_idx].pos_and_or ==
                   goal_pos_and_or or state.num_orders_remaining == 0)
        heuristic_fn = lambda state: sum(
            pos_distance(state.players[agent_idx].position, goal_pos_and_or[0]))

        search_problem = SearchTree(state,
                                    goal_fn,
                                    expand_fn,
                                    heuristic_fn,
                                    max_iter_count=1e6)
        state_action_plan, cost = search_problem.A_star_graph_search(info=False)
        action_plan, state_plan = zip(*state_action_plan)
        action_plan = action_plan[1:]
        end_state = state_plan[-1]
        return action_plan, end_state, cost

    def embedded_mdp_succ_fn(self, state, other_agent):
        other_agent_action, _ = other_agent.action(state)

        successors = []
        for a in Action.ALL_ACTIONS:
            successor_state, joint_action = self.embedded_mdp_step(
                state, a, other_agent_action, other_agent.agent_index)
            cost = 1
            successors.append((joint_action, successor_state, cost))
        return successors

    def embedded_mdp_step(self, state, action, other_agent_action,
                          other_agent_index):
        if other_agent_index == 0:
            joint_action = (other_agent_action, action)
        else:
            joint_action = (action, other_agent_action)
        if not self.mdp.is_terminal(state):
            results, _, _, _ = self.mdp.get_state_transition(
                state, joint_action)
            successor_state = results
        else:
            logger.info("Tried to find successor of terminal")
            assert False, f"state {state} \t action {action}"
            successor_state = state
        return successor_state, joint_action


class Heuristic:

    def __init__(self, mp):
        self.motion_planner = mp
        self.mdp = mp.mdp
        self.heuristic_cost_dict = self._calculate_heuristic_costs()

    def hard_heuristic(self, state, goal_deliveries, time=0, debug=False):
        # NOTE: does not support tomatoes – currently deprecated as harder
        # heuristic does not seem worth the additional computational time
        """
        From a state, we can calculate exactly how many:
        - soup deliveries we need
        - dishes to pots we need
        - onion to pots we need

        We then determine if there are any soups/dishes/onions
        in transit (on counters or on players) than can be
        brought to their destinations faster than starting off from
        a dispenser of the same type. If so, we consider fulfilling
        all demand from these positions.

        After all in-transit objects are considered, we consider the
        costs required to fulfill all the rest of the demand, that is
        given by:
        - pot-delivery trips
        - dish-pot trips
        - onion-pot trips

        The total cost is obtained by determining an optimistic time
        cost for each of these trip types
        """
        forward_cost = 0

        # Obtaining useful quantities
        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        min_pot_delivery_cost = self.heuristic_cost_dict["pot-delivery"]
        min_dish_to_pot_cost = self.heuristic_cost_dict["dish-pot"]
        min_onion_to_pot_cost = self.heuristic_cost_dict["onion-pot"]

        pot_locations = self.mdp.get_pot_locations()
        full_soups_in_pots = (pot_states_dict["onion"]["cooking"] +
                              pot_states_dict["tomato"]["cooking"] +
                              pot_states_dict["onion"]["ready"] +
                              pot_states_dict["tomato"]["ready"])
        partially_full_soups = (pot_states_dict["onion"]["partially_full"] +
                                pot_states_dict["tomato"]["partially_full"])
        num_onions_in_partially_full_pots = sum(
            [state.get_object(loc).state[1] for loc in partially_full_soups])

        # Calculating costs
        num_deliveries_to_go = goal_deliveries - state.num_delivered

        # SOUP COSTS
        total_num_soups_needed = max([0, num_deliveries_to_go])

        soups_on_counters = [
            soup_obj for soup_obj in objects_dict["soup"]
            if soup_obj.position not in pot_locations
        ]
        soups_in_transit = player_objects["soup"] + soups_on_counters
        soup_delivery_locations = self.mdp.get_serving_locations()

        (
            num_soups_better_than_pot,
            total_better_than_pot_soup_cost,
        ) = self.get_costs_better_than_dispenser(
            soups_in_transit,
            soup_delivery_locations,
            min_pot_delivery_cost,
            total_num_soups_needed,
            state,
        )

        min_pot_to_delivery_trips = max(
            [0, total_num_soups_needed - num_soups_better_than_pot])
        pot_to_delivery_costs = (min_pot_delivery_cost *
                                 min_pot_to_delivery_trips)

        forward_cost += total_better_than_pot_soup_cost
        forward_cost += pot_to_delivery_costs

        # DISH COSTS
        total_num_dishes_needed = max([0, min_pot_to_delivery_trips])
        dishes_on_counters = objects_dict["dish"]
        dishes_in_transit = player_objects["dish"] + dishes_on_counters

        (
            num_dishes_better_than_disp,
            total_better_than_disp_dish_cost,
        ) = self.get_costs_better_than_dispenser(
            dishes_in_transit,
            pot_locations,
            min_dish_to_pot_cost,
            total_num_dishes_needed,
            state,
        )

        min_dish_to_pot_trips = max(
            [0, min_pot_to_delivery_trips - num_dishes_better_than_disp])
        dish_to_pot_costs = min_dish_to_pot_cost * min_dish_to_pot_trips

        forward_cost += total_better_than_disp_dish_cost
        forward_cost += dish_to_pot_costs

        # ONION COSTS
        num_pots_to_be_filled = min_pot_to_delivery_trips - len(
            full_soups_in_pots)
        total_num_onions_needed = (num_pots_to_be_filled * 3 -
                                   num_onions_in_partially_full_pots)
        onions_on_counters = objects_dict["onion"]
        onions_in_transit = player_objects["onion"] + onions_on_counters

        (
            num_onions_better_than_disp,
            total_better_than_disp_onion_cost,
        ) = self.get_costs_better_than_dispenser(
            onions_in_transit,
            pot_locations,
            min_onion_to_pot_cost,
            total_num_onions_needed,
            state,
        )

        min_onion_to_pot_trips = max(
            [0, total_num_onions_needed - num_onions_better_than_disp])
        onion_to_pot_costs = min_onion_to_pot_cost * min_onion_to_pot_trips

        forward_cost += total_better_than_disp_onion_cost
        forward_cost += onion_to_pot_costs

        # Going to closest feature costs
        # NOTE: as implemented makes heuristic inconsistent
        # for player in state.players:
        #     if not player.has_object():
        #         counter_objects = (soups_on_counters + dishes_on_counters +
        #                            onions_on_counters)
        #         possible_features = (counter_objects + pot_locations +
        #                              self.mdp.get_dish_dispenser_locations() +
        #                              self.mdp.get_onion_dispenser_locations())
        #         forward_cost += self.action_manager.min_cost_to_feature(
        #             player.pos_and_or, possible_features)

        heuristic_cost = forward_cost / 2

        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            logger.info("\n" + "#" * 35)
            logger.info(f"Current state: (ml timestep {time})\n")

            logger.info(
                f"# in transit: \t\t Soups {len(soups_in_transit)} \t Dishes "
                f"{len(dishes_in_transit)} \t Onions {len(onions_in_transit)}")

            # NOTE Possible improvement: consider cost of dish delivery too when
            # considering if a transit soup is better than dispenser equivalent
            logger.info(
                f"# better than disp: \t Soups {num_soups_better_than_pot} \t "
                f"Dishes {num_dishes_better_than_disp} \t Onions "
                f"{num_onions_better_than_disp}")

            logger.info(
                f"# of trips: \t\t pot-del {min_pot_to_delivery_trips} \t "
                f"dish-pot {min_dish_to_pot_trips} \t onion-pot "
                f"{min_onion_to_pot_trips}")

            logger.info(
                f"Trip costs: \t\t pot-del {pot_to_delivery_costs} \t dish-pot "
                f"{dish_to_pot_costs} \t onion-pot {onion_to_pot_costs}")

            logger.info(str(env) + f"HEURISTIC: {heuristic_cost}")

        return heuristic_cost

    def get_costs_better_than_dispenser(
        self,
        possible_objects,
        target_locations,
        baseline_cost,
        num_needed,
        state,
    ):
        """
        Computes the number of objects whose minimum cost to any of the target
        locations is smaller than the baseline cost (clipping it if greater than
        the number needed). It also calculates a lower bound on the cost of
        using such objects.
        """
        costs_from_transit_locations = []
        for obj in possible_objects:
            obj_pos = obj.position
            if obj_pos in state.player_positions:
                # If object is being carried by a player
                player = [p for p in state.players if p.position == obj_pos][0]
                # NOTE: not sure if this -1 is justified.
                # Made things work better in practice for greedy heuristic based
                # agents.
                # For now this function is just used from there. Consider
                # removing later if greedy heuristic agents end up not being
                # used.
                min_cost = (self.motion_planner.min_cost_to_feature(
                    player.pos_and_or, target_locations) - 1)
            else:
                # If object is on a counter
                min_cost = self.motion_planner.min_cost_between_features(
                    [obj_pos], target_locations)
            costs_from_transit_locations.append(min_cost)

        costs_better_than_dispenser = [
            cost for cost in costs_from_transit_locations
            if cost <= baseline_cost
        ]
        better_than_dispenser_total_cost = sum(
            np.sort(costs_better_than_dispenser)[:num_needed])
        return (
            len(costs_better_than_dispenser),
            better_than_dispenser_total_cost,
        )

    def _calculate_heuristic_costs(self, debug=False):
        """Pre-computes the costs between common trip types for this mdp"""
        pot_locations = self.mdp.get_pot_locations()
        delivery_locations = self.mdp.get_serving_locations()
        dish_locations = self.mdp.get_dish_dispenser_locations()
        onion_locations = self.mdp.get_onion_dispenser_locations()
        tomato_locations = self.mdp.get_tomato_dispenser_locations()

        heuristic_cost_dict = {
            "pot-delivery":
                self.motion_planner.min_cost_between_features(
                    pot_locations, delivery_locations, manhattan_if_fail=True),
            "dish-pot":
                self.motion_planner.min_cost_between_features(
                    dish_locations, pot_locations, manhattan_if_fail=True),
        }

        onion_pot_cost = self.motion_planner.min_cost_between_features(
            onion_locations, pot_locations, manhattan_if_fail=True)
        tomato_pot_cost = self.motion_planner.min_cost_between_features(
            tomato_locations, pot_locations, manhattan_if_fail=True)

        if debug:
            logger.info(f"Heuristic cost dict {heuristic_cost_dict}")
        assert onion_pot_cost != np.inf or tomato_pot_cost != np.inf
        if onion_pot_cost != np.inf:
            heuristic_cost_dict["onion-pot"] = onion_pot_cost
        if tomato_pot_cost != np.inf:
            heuristic_cost_dict["tomato-pot"] = tomato_pot_cost

        return heuristic_cost_dict

    def simple_heuristic(self, state, time=0, debug=False):
        """Simpler heuristic that tends to run faster than current one"""
        # NOTE: State should be modified to have an order list w.r.t. which
        # one can calculate the heuristic
        assert state.order_list is not None

        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        num_deliveries_to_go = state.num_orders_remaining

        full_soups_in_pots = (pot_states_dict["onion"]["cooking"] +
                              pot_states_dict["tomato"]["cooking"] +
                              pot_states_dict["onion"]["ready"] +
                              pot_states_dict["tomato"]["ready"])
        partially_full_onion_soups = pot_states_dict["onion"]["partially_full"]
        partially_full_tomato_soups = pot_states_dict["tomato"][
            "partially_full"]
        num_onions_in_partially_full_pots = sum([
            state.get_object(loc).state[1] for loc in partially_full_onion_soups
        ])
        num_tomatoes_in_partially_full_pots = sum([
            state.get_object(loc).state[1]
            for loc in partially_full_tomato_soups
        ])

        soups_in_transit = player_objects["soup"]
        dishes_in_transit = objects_dict["dish"] + player_objects["dish"]
        onions_in_transit = objects_dict["onion"] + player_objects["onion"]
        tomatoes_in_transit = objects_dict["tomato"] + player_objects["tomato"]

        num_pot_to_delivery = max(
            [0, num_deliveries_to_go - len(soups_in_transit)])
        num_dish_to_pot = max([0, num_pot_to_delivery - len(dishes_in_transit)])

        num_pots_to_be_filled = num_pot_to_delivery - len(full_soups_in_pots)
        num_onions_needed_for_pots = (num_pots_to_be_filled * 3 -
                                      len(onions_in_transit) -
                                      num_onions_in_partially_full_pots)
        num_tomatoes_needed_for_pots = (num_pots_to_be_filled * 3 -
                                        len(tomatoes_in_transit) -
                                        num_tomatoes_in_partially_full_pots)
        num_onion_to_pot = max([0, num_onions_needed_for_pots])
        num_tomato_to_pot = max([0, num_tomatoes_needed_for_pots])

        pot_to_delivery_costs = (self.heuristic_cost_dict["pot-delivery"] *
                                 num_pot_to_delivery)
        dish_to_pot_costs = (self.heuristic_cost_dict["dish-pot"] *
                             num_dish_to_pot)

        items_to_pot_costs = []
        if "onion-pot" in self.heuristic_cost_dict.keys():
            onion_to_pot_costs = (self.heuristic_cost_dict["onion-pot"] *
                                  num_onion_to_pot)
            items_to_pot_costs.append(onion_to_pot_costs)
        if "tomato-pot" in self.heuristic_cost_dict.keys():
            tomato_to_pot_costs = (self.heuristic_cost_dict["tomato-pot"] *
                                   num_tomato_to_pot)
            items_to_pot_costs.append(tomato_to_pot_costs)

        # NOTE: doesn't take into account that a combination of the two might
        # actually be more advantageous.
        # Might cause heuristic to be inadmissible in some edge cases.
        items_to_pot_cost = min(items_to_pot_costs)

        heuristic_cost = (pot_to_delivery_costs + dish_to_pot_costs +
                          items_to_pot_cost) / 2

        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            logger.info("\n" + "#" * 35)
            logger.info(f"Current state: (ml timestep {time})\n")

            logger.info(
                f"# in transit: \t\t Soups {len(soups_in_transit)} \t Dishes "
                f"{len(dishes_in_transit)} \t Onions {len(onions_in_transit)}")

            logger.info(
                f"Trip costs: \t\t pot-del {pot_to_delivery_costs} \t dish-pot "
                f"{dish_to_pot_costs} \t onion-pot {onion_to_pot_costs}")

            logger.info(str(env) + f"HEURISTIC: {heuristic_cost}")

        return heuristic_cost


class MediumLevelMdpPlanner:

    def __init__(
        self,
        mdp,
        mlp_params,
        state_dict={},
        state_idx_dict={},
        action_dict={},
        action_idx_dict={},
        transition_matrix=None,
        reward_matrix=None,
        policy_matrix=None,
        value_matrix=None,
        num_states=0,
        num_rounds=0,
        epsilon=0.01,
        discount=0.8,
    ):

        self.mdp = mdp
        self.params = mlp_params
        self.jmp = JointMotionPlanner(mdp, mlp_params)
        self.mp = self.jmp.motion_planner

        self.state_idx_dict = state_idx_dict
        self.state_dict = state_dict
        self.action_dict = action_dict
        self.action_idx_dict = action_idx_dict
        # set states as 'player's object + medium level actions (get, place,
        # deliver, put in pot)

        self.num_joint_action = Action.NUM_ACTIONS  # * Action.NUM_ACTIONS)
        self.num_states = len(state_idx_dict)
        self.num_actions = len(action_idx_dict)
        self.num_rounds = num_rounds
        self.planner_name = "mdp"
        self.agent_index = 0

        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.policy_matrix = policy_matrix
        self.value_matrix = value_matrix
        self.epsilon = epsilon
        self.discount = discount
        self.q = None

    @staticmethod
    def from_mdp_planner_file(filename):
        with open(os.path.join(PLANNERS_DIR, filename), "rb") as f:
            mdp_planner = pickle.load(f)
            mdp = mdp_planner[0]
            params = mdp_planner[1]

            state_idx_dict = mdp_planner[2]
            state_dict = mdp_planner[3]

            # transition_matrix = mdp_planner.transition_matrix
            # reward_matrix = mdp_planner.reward_matrix
            policy_matrix = mdp_planner[4]
            # value_matrix = mdp_planner.value_matrix

            # num_states = mdp_planner.num_states
            # num_rounds = mdp_planner.num_rounds

            return MediumLevelMdpPlanner(
                mdp,
                params,
                state_dict,
                state_idx_dict,
                policy_matrix=policy_matrix,
            )

    @staticmethod
    def from_pickle_or_compute(
        mdp,
        mlp_params,
        custom_filename=None,
        force_compute_all=False,
        info=True,
        force_compute_more=False,
    ):

        assert isinstance(mdp, OvercookedGridworld)

        filename = (custom_filename if custom_filename is not None else
                    mdp.layout_name + "_" + "medium_mdp" + ".pkl")

        if force_compute_all:
            mdp_planner = MediumLevelMdpPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(filename)
            return mdp_planner

        try:
            mdp_planner = MediumLevelMdpPlanner.from_mdp_planner_file(filename)

            if force_compute_more:
                logger.info(
                    "Stored mdp_planner computed ",
                    str(mdp_planner.num_rounds),
                    " rounds. Compute another " + str(TRAINNINGUNIT) +
                    " more...",
                )
                mdp_planner.compute_mdp_policy(filename)
                return mdp_planner

        except (
                FileNotFoundError,
                ModuleNotFoundError,
                EOFError,
                AttributeError,
        ) as e:
            logger.info("Recomputing planner due to:", e)
            mdp_planner = MediumLevelMdpPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp_policy(filename)
            return mdp_planner

        if info:
            logger.info(f"Loaded MediumMdpPlanner from "
                        f"{os.path.join(PLANNERS_DIR, filename)}")

        return mdp_planner

    def save_policy_to_file(self, filename):
        with open(filename, "wb") as output:
            mdp_plan = [
                self.mdp,
                self.params,
                self.state_idx_dict,
                self.state_dict,
                self.action_idx_dict,
                self.action_dict,
                self.transition_matrix,
                self.policy_matrix,
            ]
            pickle.dump(mdp_plan, output, pickle.HIGHEST_PROTOCOL)

    def gen_state_dict_key(self, state, player, soup_finish, other_player=None):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, len(order_list)

        player_obj = None
        if player.held_object is not None:
            player_obj = player.held_object.name

        order_str = None if state.order_list is None else state.order_list[0]
        for order in state.order_list[1:]:
            order_str = order_str + "_" + str(order)

        state_str = str(player_obj) + "_" + str(soup_finish) + "_" + order_str

        return state_str

    def init_states(self, state_idx_dict=None, order_list=None):
        # logger.info("In init_states()...")
        # player_obj, num_item_in_pot, order_list

        if state_idx_dict is None:
            objects = ["onion", "soup", "dish", "None"]  # "tomato"
            # common_actions = ["pickup", "drop"]
            # addition_actions = [
            #     ("soup", "deliver"),
            #     ("soup", "pickup"),
            #     ("dish", "pickup"),
            #     ("None", "None"),
            # ]
            # obj_action_pair = (
            #     list(itertools.product(objects, common_actions)) +
            #     addition_actions)

            state_keys = []
            state_obj = []
            tmp_state_obj = []
            tmp_state_obj_1 = []

            for obj in objects:
                tmp_state_obj.append(([obj]))

            # include number of item in soup
            objects_only_arr = [obj.copy() for obj in tmp_state_obj]
            for i in range(self.mdp.num_items_for_soup + 1):
                tmp_keys = [val + "_" + str(i) for val in objects]
                for obj in tmp_state_obj:
                    obj.append(i)

                state_keys = state_keys + tmp_keys
                state_obj = state_obj + tmp_state_obj
                tmp_state_obj = [obj.copy() for obj in objects_only_arr]

            tmp_state_key = state_keys
            tmp_state_obj = [obj.copy() for obj in state_obj]

            # include order list items in state

            for order in order_list:
                prev_keys = tmp_state_key.copy()
                tmp_keys = [i + "_" + order for i in prev_keys]
                state_keys = state_keys + tmp_keys
                tmp_state_key = tmp_keys

                for obj in tmp_state_obj:
                    obj.append(order)
                state_obj = state_obj + [obj.copy() for obj in tmp_state_obj]

            # logger.info(state_keys, state_obj)

            self.state_idx_dict = {k: i for i, k in enumerate(state_keys)}
            self.state_dict = {
                key: obj for key, obj in zip(state_keys, state_obj)
            }

        else:
            self.state_idx_dict = state_idx_dict
            self.state_dict = state_dict  # TODO: state_dict not defined

        # logger.info("Initialize states:", self.state_idx_dict.items())
        return

    def init_actions(self, actions=None):
        # logger.info("In init_actions()...")

        if actions is None:
            objects = ["onion", "dish"]  # "tomato"
            common_actions = ["pickup", "drop"]
            addition_actions = [["deliver", "soup"], ["pickup", "soup"]]

            common_action_obj_pair = list(
                itertools.product(common_actions, objects))
            common_action_obj_pair = [list(i) for i in common_action_obj_pair]
            actions = common_action_obj_pair + addition_actions
            self.action_dict = {
                action[0] + "_" + action[1]: action for action in actions
            }
            self.action_idx_dict = {
                action[0] + "_" + action[1]: i
                for i, action in enumerate(actions)
            }

        else:
            self.action_dict = action_dict  # TODO: action_dict not defined
            self.action_idx_dict = (
                action_idx_dict  # TODO: action_idx_dict not defined
            )

        # logger.info("Initialize actions:", self.action_dict)

        return

    def init_transition_matrix(self, transition_matrix=None):
        self.transition_matrix = (transition_matrix if transition_matrix
                                  is not None else np.zeros(
                                      (
                                          len(self.action_dict),
                                          len(self.state_idx_dict),
                                          len(self.state_idx_dict),
                                      ),
                                      dtype=float,
                                  ))

        game_logic_transition = self.transition_matrix.copy()
        distance_transition = self.transition_matrix.copy()

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            for action_key, action_idx in self.action_idx_dict.items():
                state_idx = self.state_idx_dict[state_key]
                next_state_idx = state_idx
                next_action_idx = action_idx

                # define state and action game transition logic
                player_obj, soup_finish, orders = self.ml_state_to_objs(
                    state_obj)
                next_actions, next_state_keys = self.state_action_nxt_state(
                    player_obj, soup_finish, orders)

                if next_actions == action_key:
                    next_state_idx = self.state_idx_dict[next_state_keys]

                game_logic_transition[next_action_idx][state_idx][
                    next_state_idx] += 1.0

            # logger.info(state_key)
        # logger.info(game_logic_transition[:, 25])
        # tmp = input()

        self.transition_matrix = game_logic_transition

    @staticmethod
    def ml_state_to_objs(state_obj):
        # state: obj + action + bool(soup nearly finish) + orders
        player_obj = state_obj[0]
        soup_finish = state_obj[1]
        orders = []
        if len(state_obj) > 2:
            orders = state_obj[2:]

        return player_obj, soup_finish, orders

    def state_action_nxt_state(self,
                               player_obj,
                               soup_finish,
                               orders,
                               other_obj=""):
        # game logic
        actions = ""
        next_obj = player_obj
        next_soup_finish = soup_finish
        if player_obj == "None":
            if (soup_finish
                    == self.mdp.num_items_for_soup) and (other_obj != "dish"):
                actions = "pickup_dish"
                next_obj = "dish"
            else:
                next_order = None
                if len(orders) > 1:
                    next_order = orders[1]

                if next_order == "onion":
                    actions = "pickup_onion"
                    next_obj = "onion"

                elif next_order == "tomato":
                    actions = "pickup_tomato"
                    next_obj = "tomato"

                else:
                    actions = "pickup_onion"
                    next_obj = "onion"

        else:
            if player_obj == "onion":
                actions = "drop_onion"
                next_obj = "None"
                next_soup_finish += 1

            elif player_obj == "tomato":
                actions = "drop_tomato"
                next_obj = "None"
                next_soup_finish += 1

            elif (player_obj == "dish") and (soup_finish
                                             == self.mdp.num_items_for_soup):
                actions = "pickup_soup"
                next_obj = "soup"
                next_soup_finish = 0

            elif (player_obj
                  == "dish") and (soup_finish != self.mdp.num_items_for_soup):
                actions = "drop_dish"
                next_obj = "None"

            elif player_obj == "soup":
                actions = "deliver_soup"
                next_obj = "None"
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                logger.info(player_obj)
                raise ValueError()

        if next_soup_finish > self.mdp.num_items_for_soup:
            next_soup_finish = self.mdp.num_items_for_soup

        next_state_keys = next_obj + "_" + str(next_soup_finish)
        for order in orders:
            next_state_keys = next_state_keys + "_" + order

        return actions, next_state_keys

    def state_transition_by_distance(self, curr_state, next_state, action):
        action_taken = action[0]
        action_obj = action[1]
        curr_state_obj = curr_state[0]
        curr_state_action = curr_state[1]
        next_state_obj = next_state[0]
        next_state_action = next_state[1]

        # location depends on the action and object in hand
        curr_location = self.map_action_to_location(curr_state_action,
                                                    curr_state_obj)
        next_location = self.map_action_to_location(action_taken, action_obj)

        # calculate distance between locations
        min_distance = self.mp.min_cost_between_features(
            curr_location, next_location)

        return 1.0 / min_distance

    def drop_item(self, state):
        return self.mdp.get_empty_counter_locations(state)

    def map_action_to_location(self,
                               world_state,
                               state_obj,
                               action,
                               obj,
                               p0_obj=None):
        """
        Get the next location the agent will be in based on current world state
        and medium level actions.
        """
        p0_obj = p0_obj if p0_obj is not None else self.state_dict[state_obj][0]
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []
        if action == "pickup" and obj != "soup":
            if p0_obj != "None":
                location = self.drop_item(world_state)
            else:
                if obj == "onion":
                    location = self.mdp.get_onion_dispenser_locations()
                elif obj == "tomato":
                    location = self.mdp.get_tomato_dispenser_locations()
                elif obj == "dish":
                    location = self.mdp.get_dish_dispenser_locations()
                else:
                    logger.info(p0_obj, action, obj)
                    ValueError()
        elif action == "pickup" and obj == "soup":
            if p0_obj != "dish" and p0_obj != "None":
                location = self.drop_item(world_state)
            elif p0_obj == "None":
                location = self.mdp.get_dish_dispenser_locations()
            else:
                location = (self.mdp.get_ready_pots(pots_states_dict) +
                            self.mdp.get_cooking_pots(pots_states_dict) +
                            self.mdp.get_full_pots(pots_states_dict))

        elif action == "drop":
            if obj == "onion" or obj == "tomato":
                location = self.mdp.get_partially_full_pots(
                    pots_states_dict) + self.mdp.get_empty_pots(
                        pots_states_dict)
            elif obj == "dish":
                location = self.drop_item(world_state)
            else:
                logger.info(p0_obj, action, obj)
                ValueError()

        elif action == "deliver":
            if p0_obj != "soup":
                location = self.mdp.get_empty_counter_locations(world_state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            logger.info(p0_obj, action, obj)
            ValueError()

        return location

    def map_action_to_state_location(self, state, state_str, action, obj,
                                     world_info):
        pots_states_dict = self.mdp.get_pot_states(world_info)
        location = []
        if action == "pickup" and obj != "soup":
            if not self._not_holding_object(state_str):
                location = self.drop_item(state)
            else:
                if obj == "onion":
                    location = self.mdp.get_onion_dispenser_locations()
                elif obj == "tomato":
                    location = self.mdp.get_tomato_dispenser_locations()
                elif obj == "dish":
                    location = self.mdp.get_dish_dispenser_locations()
                else:
                    ValueError()
        elif action == "pickup" and obj == "soup":
            if self.state_dict[state_str][
                    0] != "dish" and not self._not_holding_object(state_str):
                location = self.drop_item(state)
            elif self._not_holding_object(state_str):
                location = self.mdp.get_dish_dispenser_locations()
            else:
                location = (
                    self.mdp.get_ready_pots(self.mdp.get_pot_states(state)) +
                    self.mdp.get_cooking_pots(self.mdp.get_pot_states(state)) +
                    self.mdp.get_full_pots(self.mdp.get_pot_states(state)))

        elif action == "drop":
            if obj == "onion" or obj == "tomato":
                location = self.mdp.get_pot_locations()
            else:
                ValueError()

        elif action == "deliver":
            if self.state_dict[state_str][0] != "soup":
                location = self.mdp.get_empty_counter_locations(state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            ValueError()

        return location

    def _not_holding_object(self, state_obj):
        return self.state_dict[state_obj][0] == "None"

    def init_reward(self, reward_matrix=None):
        # state: obj + action + bool(soup nearly finish) + orders

        self.reward_matrix = (
            reward_matrix if reward_matrix is not None else np.zeros(
                (len(self.action_dict), len(self.state_idx_dict)), dtype=float))

        # when deliver order, pickup onion. probabily checking the change in
        # states to give out rewards: if action is correct, curr_state acts and
        # changes to rewardable next state. Then, we reward.

        for state_key, state_obj in self.state_dict.items():
            # state: obj + action + bool(soup nearly finish) + orders
            player_obj = state_obj[0]
            soup_finish = state_obj[1]
            orders = []
            if len(state_obj) > 2:
                orders = state_obj[2:]

            if player_obj == "soup":
                self.reward_matrix[self.action_idx_dict["deliver_soup"]][
                    self.state_idx_dict[state_key]] += self.mdp.delivery_reward

            if len(orders) == 0:
                self.reward_matrix[:, self.state_idx_dict[
                    state_key]] += self.mdp.delivery_reward

            # if (soup_finish == self.mdp.num_items_for_soup and
            #         player_obj == "dish"):
            #     self.reward_matrix[self.action_idx_dict["pickup_soup"],
            #                        self.state_idx_dict[state_key],] += (
            #                            self.mdp.delivery_reward / 5.0)

    def bellman_operator(self, V=None):

        if V is None:
            V = self.value_matrix

        Q = np.zeros((self.num_actions, self.num_states))
        for a in range(self.num_actions):
            # logger.info(self.transition_matrix[a].dot(V))
            Q[a] = self.reward_matrix[
                a] + self.discount * self.transition_matrix[a].dot(V)

        return Q.max(axis=0), Q.argmax(axis=0)

    @staticmethod
    def get_span(arr):
        # logger.info(
        #     "in get span arr.max():",
        #     arr.max(),
        #     " - arr.min():",
        #     arr.min(),
        #     " = ",
        #     (arr.max() - arr.min()),
        # )
        return arr.max() - arr.min()

    def log_value_iter(self, iter_count):
        self.num_rounds = iter_count
        output_filename = (self.mdp.layout_name + "_" + self.planner_name +
                           "_" + str(self.num_rounds) + ".pkl")
        output_mdp_path = os.path.join(PLANNERS_DIR, output_filename)
        self.save_policy_to_file(output_mdp_path)

        return

    def value_iteration(self, value_matrix=None):
        self.value_matrix = (value_matrix if value_matrix is not None else
                             np.zeros(self.num_states, dtype=float))
        self.policy_matrix = (value_matrix if value_matrix is not None else
                              np.zeros(self.num_states, dtype=float))

        # computation of threshold of variation for V for an epsilon-optimal
        # policy
        if self.discount < 1.0:
            thresh = self.epsilon * (1 - self.discount) / self.discount
        else:
            thresh = self.epsilon

        iter_count = 0
        while True:
            V_prev = self.value_matrix.copy()

            self.value_matrix, self.policy_matrix = self.bellman_operator()

            variation = self.get_span(self.value_matrix - V_prev)
            # logger.info(self.value_matrix)
            # logger.info("Variation =",  variation, ", Threshold =", thresh)

            if variation < thresh:
                self.log_value_iter(iter_count)
                break
            elif iter_count % LOGUNIT == 0:
                self.log_value_iter(iter_count)
            else:
                pass

            iter_count += 1

        return

    def save_to_file(self, filename):
        logger.info("In save_to_file")
        with open(filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def init_mdp(self):
        self.init_states(order_list=self.mdp.start_order_list)
        self.init_actions()
        self.init_transition_matrix()
        self.init_reward()

    def compute_mdp_policy(self, filename):
        # start_time = time.time()

        final_filepath = os.path.join(PLANNERS_DIR, filename)
        self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)
        # logger.info(
        #     "Total states =",
        #     self.num_states,
        #     "; Total actions =",
        #     self.num_actions,
        # )

        self.value_iteration()

        # logger.info("Policy Probability Distribution = ")
        # logger.info(self.policy_matrix.tolist(), '\n')
        # logger.info(self.policy_matrix.shape)

        # logger.info("without GPU:", timer()-start)
        # logger.info(f"It took {time.time() - start_time} seconds to create "
        #             f"MediumLevelMdpPlanner")
        # self.save_to_file(final_filepath)
        # tmp = input()
        # self.save_to_file(output_mdp_path)
        return


class HumanSubtaskQMDPPlanner(MediumLevelMdpPlanner):

    def __init__(
        self,
        mdp,
        mlp_params,
        state_dict={},
        state_idx_dict={},
        action_dict={},
        action_idx_dict={},
        transition_matrix=None,
        reward_matrix=None,
        policy_matrix=None,
        value_matrix=None,
        num_states=0,
        num_rounds=0,
        epsilon=0.01,
        discount=0.8,
    ):

        super().__init__(
            mdp,
            mlp_params,
            state_dict={},
            state_idx_dict={},
            action_dict={},
            action_idx_dict={},
            transition_matrix=None,
            reward_matrix=None,
            policy_matrix=None,
            value_matrix=None,
            num_states=0,
            num_rounds=0,
            epsilon=0.01,
            discount=0.8,
        )

        self.world_state_cost_dict = {}
        self.jmp = JointMotionPlanner(mdp, mlp_params)
        self.mp = self.jmp.motion_planner
        self.subtask_dict = {}
        self.subtask_idx_dict = {}

    @staticmethod
    def from_pickle_or_compute(
        mdp,
        mlp_params,
        custom_filename=None,
        force_compute_all=False,
        info=True,
        force_compute_more=False,
    ):

        assert isinstance(mdp, OvercookedGridworld)

        filename = (custom_filename if custom_filename is not None else
                    mdp.layout_name + "_" + "human_subtask_aware_qmdp" + ".pkl")

        if force_compute_all:
            mdp_planner = HumanSubtaskQMDPPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp(filename)
            return mdp_planner

        try:
            mdp_planner = HumanSubtaskQMDPPlanner.from_mdp_planner_file(
                filename)

            if force_compute_more:
                logger.info(
                    "Stored mdp_planner computed ",
                    str(mdp_planner.num_rounds),
                    " rounds. Compute another " + str(TRAINNINGUNIT) +
                    " more...",
                )
                # TODO: compute_mdp not defined
                mdp_planner.compute_mdp(filename)
                return mdp_planner

        except (
                FileNotFoundError,
                ModuleNotFoundError,
                EOFError,
                AttributeError,
        ) as e:
            logger.info("Recomputing planner due to:", e)
            mdp_planner = HumanSubtaskQMDPPlanner(mdp, mlp_params)
            mdp_planner.compute_mdp(filename)
            return mdp_planner

        if info:
            logger.info(f"Loaded HumanSubtaskQMDPPlanner from "
                        f"{os.path.join(PLANNERS_DIR, filename)}")

        return mdp_planner

    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        """
        States: agent 0 holding object, number of item in pot, order list, agent
        1 (usually human) holding object, agent 1 subtask.
        """

        # set state dict as [p0_obj, num_item_in_pot, order_list]
        self.init_states(order_list=order_list)

        # add [p1_obj, subtask] to [p0_obj, num_item_in_pot, order_list]
        objects = ["onion", "soup", "dish", "None"]  # "tomato"
        self.subtask_dict = copy.deepcopy(self.action_dict)
        del self.subtask_dict["drop_dish"]
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()
        for i, obj in enumerate(objects):
            for j, subtask in enumerate(self.subtask_dict.items()):
                self.subtask_idx_dict[subtask[0]] = j
                if self._init_is_valid_object_subtask_pair(obj, subtask[0]):
                    for ori_key, ori_value in original_state_dict.items():
                        new_key = ori_key + "_" + obj + "_" + subtask[0]
                        new_obj = (original_state_dict[ori_key] + [obj] +
                                   [subtask[0]])
                        self.state_dict[new_key] = new_obj  # update value
                        self.state_idx_dict[new_key] = len(self.state_idx_dict)

    def init_transition(self, transition_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both
        robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.transition_matrix = (transition_matrix if transition_matrix
                                  is not None else np.zeros(
                                      (
                                          len(self.action_dict),
                                          len(self.state_idx_dict),
                                          len(self.state_idx_dict),
                                      ),
                                      dtype=float,
                                  ))

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            for action_key, action_idx in self.action_idx_dict.items():
                normalize_count = 0

                # decode state information
                p0_state, p1_state, world_info = self.decode_state_info(
                    state_obj
                )  # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
                # calculate next states for p1 (a.k.a. human)
                (
                    p1_nxt_states,
                    p1_nxt_world_info,
                ) = self.human_state_subtask_transition(p1_state, world_info)

                # calculate next states for p0 (conditioned on p1 (a.k.a.
                # human))
                for p1_nxt_state in p1_nxt_states:
                    action, next_state_key = self.state_transition(
                        p0_state, p1_nxt_world_info, human_state=p1_nxt_state)
                    # for action, next_state_key in zip(actions, next_state_keys):
                    #     logger.info(
                    #         p0_state,
                    #         p1_nxt_world_info,
                    #         p1_nxt_state,
                    #         action,
                    #         next_state_keys,
                    #     )
                    if action_key == action:
                        next_state_idx = self.state_idx_dict[next_state_key]
                        self.transition_matrix[action_idx, state_idx,
                                               next_state_idx] += 1.0

                if np.sum(self.transition_matrix[action_idx, state_idx]) > 0.0:
                    self.transition_matrix[action_idx, state_idx] /= np.sum(
                        self.transition_matrix[action_idx, state_idx])

        self.transition_matrix[self.transition_matrix == 0.0] = 0.000001

    def get_successor_states(self,
                             start_world_state,
                             start_state_key,
                             debug=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if (len(self.state_dict[start_state_key][2:]) <=
                2):  # [p0_obj, num_item_in_soup, orders, p1_obj, subtask]
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        agent_action_idx_arr, next_state_idx_arr = np.where(
            self.transition_matrix[:, ori_state_idx] >
            0.000001)  # returns array(action idx), array(next_state_idx)
        start_time = time.time()
        for next_action_idx, next_state_idx in zip(agent_action_idx_arr,
                                                   next_state_idx_arr):
            next_world_state, cost = self.mdp_action_state_to_world_state(
                next_action_idx, next_state_idx, start_world_state)
            successor_states.append((
                self.get_key_from_value(self.state_idx_dict, next_state_idx),
                next_world_state,
                cost,
            ))
            if debug:
                logger.info(
                    f"Action {self.get_key_from_value(self.action_idx_dict, next_action_idx)} "
                    f"from {self.get_key_from_value(self.state_idx_dict, ori_state_idx)} "
                    f"to {self.get_key_from_value(self.state_idx_dict, next_state_idx)} "
                    f"costs {cost} in {time.time() - start_time} seconds.")

        return successor_states

    @staticmethod
    def decode_state_info(state_obj):
        return state_obj[0], state_obj[-2:], state_obj[1:-2]

    @staticmethod
    def _init_is_valid_object_subtask_pair(obj, subtask):
        if obj == "None":
            if subtask == "pickup_dish":
                return True
            elif subtask == "pickup_onion":
                return True
            elif subtask == "pickup_tomato":
                return True
            # elif subtask == "pickup_soup":
            #     return True
            else:
                return False
        else:
            if obj == "onion" and subtask == "drop_onion":
                return True
            elif obj == "tomato" and subtask == "drop_tomato":
                return True
            elif (obj == "dish") and subtask == "pickup_soup":
                return True
            # elif (obj == "dish") and subtask == "drop_dish":
            #     return True
            elif obj == "soup" and subtask == "deliver_soup":
                return True
            else:
                return False
        return True

    def _is_valid_object_subtask_pair(self,
                                      obj,
                                      subtask,
                                      soup_finish,
                                      greedy=False):
        if obj == "None":
            if (not greedy and
                (subtask == "pickup_dish" or subtask == "pickup_onion") and
                    soup_finish <= self.mdp.num_items_for_soup):
                return True
            elif (greedy and subtask == "pickup_onion" and
                  soup_finish < self.mdp.num_items_for_soup):
                return True
            elif (greedy and subtask == "pickup_dish" and
                  soup_finish == self.mdp.num_items_for_soup):
                return True
            elif subtask == "pickup_tomato":
                return True
            # elif subtask == "pickup_soup":
            #     return True
            else:
                return False
        else:
            if obj == "onion" and subtask == "drop_onion":
                return True
            elif obj == "tomato" and subtask == "drop_tomato":
                return True
            elif (obj == "dish") and subtask == "pickup_soup":
                return True
            # elif (obj == "dish") and subtask == "drop_dish":
            #     return True
            elif obj == "soup" and subtask == "deliver_soup":
                return True
            else:
                return False
        return True

    def human_state_subtask_transition(self, human_state, world_info):
        player_obj = human_state[0]
        subtask = human_state[1]
        soup_finish = world_info[0]
        orders = [] if len(world_info) < 2 else world_info[1:]
        next_obj = player_obj
        next_subtasks = []
        next_soup_finish = soup_finish

        if player_obj == "None":
            if subtask == "pickup_dish":
                next_obj = "dish"
                next_subtasks = ["pickup_soup"]  # , "drop_dish"]

            elif subtask == "pickup_onion":
                next_obj = "onion"
                next_subtasks = ["drop_onion"]

            elif subtask == "pickup_tomato":
                next_obj = "tomato"
                next_subtasks = ["drop_tomato"]

            # elif subtask == "pickup_soup":
            #     next_obj = "soup"
            #     next_subtasks = ["deliver_soup"]

        else:
            if (player_obj == "onion" and subtask == "drop_onion" and
                    soup_finish < self.mdp.num_items_for_soup):
                next_obj = "None"
                next_soup_finish += 1
                next_subtasks = [
                    "pickup_onion",
                    "pickup_dish",
                ]  # "pickup_tomato"

            elif (player_obj == "onion" and subtask == "drop_onion" and
                  soup_finish == self.mdp.num_items_for_soup):
                next_obj = "onion"
                next_subtasks = ["drop_onion"]

            elif player_obj == "tomato" and subtask == "drop_tomato":
                next_obj = "None"
                next_soup_finish += 1
                next_subtasks = [
                    "pickup_onion",
                    "pickup_dish",
                ]  # "pickup_tomato"

            elif (player_obj == "dish") and subtask == "pickup_soup":
                next_obj = "soup"
                next_soup_finish = 0
                next_subtasks = ["deliver_soup"]

            # elif (player_obj == "dish") and subtask == "drop_dish":
            #     next_obj = "None"
            #     next_subtasks = [
            #         "pickup_onion",
            #         "pickup_dish",
            #     ]  # "pickup_tomato"

            elif player_obj == "soup" and subtask == "deliver_soup":
                next_obj = "None"
                next_subtasks = [
                    "pickup_onion",
                    "pickup_dish",
                ]  # "pickup_tomato"
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                logger.info(player_obj, subtask)
                raise ValueError()

        if next_soup_finish > self.mdp.num_items_for_soup:
            next_soup_finish = self.mdp.num_items_for_soup

        p1_nxt_states = []
        for next_subtask in next_subtasks:
            p1_nxt_states.append([next_obj, next_subtask])

        nxt_world_info = [next_soup_finish]
        for order in orders:
            nxt_world_info.append(order)

        return p1_nxt_states, nxt_world_info

    def state_transition(self,
                         player_obj,
                         world_info,
                         human_state=[None, None]):
        # game logic
        soup_finish = world_info[0]
        orders = [] if len(world_info) < 2 else world_info[1:]
        other_obj = human_state[0]
        subtask = human_state[1]
        actions = ""
        next_obj = player_obj
        next_soup_finish = soup_finish

        if player_obj == "None":
            if (soup_finish == self.mdp.num_items_for_soup) and (
                    other_obj != "dish" and subtask != "pickup_dish"):
                actions = "pickup_dish"
                next_obj = "dish"
            elif (soup_finish == (self.mdp.num_items_for_soup - 1)) and (
                    other_obj == "onion" and subtask == "drop_onion"):
                actions = "pickup_dish"
                next_obj = "dish"
            else:
                next_order = None
                if len(orders) > 1:
                    next_order = orders[1]

                if next_order == "onion":
                    actions = "pickup_onion"
                    next_obj = "onion"

                elif next_order == "tomato":
                    actions = "pickup_tomato"
                    next_obj = "tomato"

                else:
                    actions = "pickup_onion"
                    next_obj = "onion"

        else:
            if player_obj == "onion":
                actions = "drop_onion"
                next_obj = "None"
                next_soup_finish += 1

            elif player_obj == "tomato":
                actions = "drop_tomato"
                next_obj = "None"
                next_soup_finish += 1

            elif (player_obj == "dish") and (soup_finish >=
                                             self.mdp.num_items_for_soup - 1):
                actions = "pickup_soup"
                next_obj = "soup"
                next_soup_finish = 0

            elif (player_obj == "dish") and (soup_finish <
                                             self.mdp.num_items_for_soup - 1):
                actions = "drop_dish"
                next_obj = "None"

            elif player_obj == "soup":
                actions = "deliver_soup"
                next_obj = "None"
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                logger.info(player_obj)
                raise ValueError()

        if next_soup_finish > self.mdp.num_items_for_soup:
            next_soup_finish = self.mdp.num_items_for_soup

        next_state_keys = next_obj + "_" + str(next_soup_finish)

        for order in orders:
            next_state_keys = next_state_keys + "_" + order

        for human_info in human_state:
            next_state_keys = next_state_keys + "_" + human_info

        return actions, next_state_keys

    @staticmethod
    def world_state_to_mdp_state_key(state, player, other_player, subtask):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, len(order_list)

        player_obj = None
        other_player_obj = None
        if player.held_object is not None:
            player_obj = player.held_object.name
        if other_player.held_object is not None:
            other_player_obj = other_player.held_object.name

        order_str = None if len(state.order_list) == 0 else state.order_list[0]
        for order in state.order_list[1:]:
            order_str = order_str + "_" + str(order)

        num_item_in_pot = 0
        if state.objects is not None and len(state.objects) > 0:
            for obj_pos, obj_state in state.objects.items():
                if (obj_state.name == "soup" and
                        obj_state.state[1] > num_item_in_pot):
                    num_item_in_pot = obj_state.state[1]

        state_strs = (str(player_obj) + "_" + str(num_item_in_pot) + "_" +
                      order_str + "_" + str(other_player_obj) + "_" + subtask)

        return state_strs

    def get_mdp_state_idx(self, mdp_state_key):
        if mdp_state_key not in self.state_dict:
            return None
        else:
            return self.state_idx_dict[mdp_state_key]

    def gen_state_dict_key(self, p0_obj, p1_obj, num_item_in_pot, orders,
                           subtasks):
        # a0 hold, a1 hold,

        player_obj = p0_obj if p0_obj is not None else "None"
        other_player_obj = p1_obj if p1_obj is not None else "None"

        order_str = None if orders is None else orders
        for order in orders:
            order_str = order_str + "_" + str(order)

        state_strs = []
        for subtask in subtasks:
            # TODO: soup_finish not defined
            state_strs.append(
                str(player_obj) + "_" + str(soup_finish) + "_" + order_str +
                "_" + str(other_player_obj) + "_" + subtask)

        return state_strs

    @staticmethod
    def get_key_from_value(dictionary, state_value):
        try:
            idx = list(dictionary.values()).index(state_value)
        except ValueError:
            return None
        else:
            return list(dictionary.keys())[idx]

    def map_action_to_location(self,
                               world_state,
                               state_str,
                               action,
                               obj,
                               p0_obj=None,
                               player_idx=None):
        """
        Get the next location the agent will be in based on current world state,
        medium level actions, after-action state obj.
        """

        p0_obj = p0_obj if p0_obj is not None else self.state_dict[state_str][0]
        other_obj = (world_state.players[1 - player_idx].held_object.name
                     if world_state.players[1 - player_idx].held_object
                     is not None else "None")
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []
        wait = False  # If wait becomes true, one player has to wait for the other player to finish its current task and its next task

        if action == "pickup" and obj != "soup":
            if p0_obj != "None":
                location = self.drop_item(world_state)
            else:
                if obj == "onion":
                    location = self.mdp.get_onion_dispenser_locations()
                elif obj == "tomato":
                    location = self.mdp.get_tomato_dispenser_locations()
                elif obj == "dish":
                    location = self.mdp.get_dish_dispenser_locations()
                else:
                    logger.info(p0_obj, action, obj)
                    ValueError()
        elif action == "pickup" and obj == "soup":
            if p0_obj != "dish" and p0_obj != "None":
                location = self.drop_item(world_state)
            elif p0_obj == "None":
                location = self.mdp.get_dish_dispenser_locations()
            else:
                if state_str is not None:
                    num_item_in_pot = self.state_dict[state_str][1]
                    if num_item_in_pot == 0:
                        location = self.mdp.get_empty_pots(pots_states_dict)
                        if len(location) > 0:
                            return location, True
                    elif 0 < num_item_in_pot < self.mdp.num_items_for_soup:
                        location = self.mdp.get_partially_full_pots(
                            pots_states_dict)
                        if len(location) > 0:
                            return location, True
                    else:
                        location = (
                            self.mdp.get_ready_pots(pots_states_dict) +
                            self.mdp.get_cooking_pots(pots_states_dict) +
                            self.mdp.get_full_pots(pots_states_dict))
                    if len(location) > 0:
                        return location, wait

                location = (self.mdp.get_ready_pots(pots_states_dict) +
                            self.mdp.get_cooking_pots(pots_states_dict) +
                            self.mdp.get_full_pots(pots_states_dict))
                if len(location) == 0:
                    wait = True
                    location = self.mdp.get_partially_full_pots(
                        pots_states_dict) + self.mdp.get_empty_pots(
                            pots_states_dict)
                    # location = world_state.players[player_idx].pos_and_or
                    return location, wait

        elif action == "drop":
            if obj == "onion" or obj == "tomato":

                if state_str is not None:
                    num_item_in_pot = self.state_dict[state_str][1]
                    if num_item_in_pot == 0:
                        location = self.mdp.get_empty_pots(pots_states_dict)
                    elif 0 < num_item_in_pot < self.mdp.num_items_for_soup:
                        location = self.mdp.get_partially_full_pots(
                            pots_states_dict)
                    else:
                        location = (
                            self.mdp.get_ready_pots(pots_states_dict) +
                            self.mdp.get_cooking_pots(pots_states_dict) +
                            self.mdp.get_full_pots(pots_states_dict))

                    if len(location) > 0:
                        return location, wait

                location = self.mdp.get_partially_full_pots(
                    pots_states_dict) + self.mdp.get_empty_pots(
                        pots_states_dict)

                if len(location) == 0:
                    if other_obj != "onion" and other_obj != "tomato":
                        wait = True
                        location = (
                            self.mdp.get_ready_pots(pots_states_dict) +
                            self.mdp.get_cooking_pots(pots_states_dict) +
                            self.mdp.get_full_pots(pots_states_dict))
                        # location = world_state.players[player_idx].pos_and_or
                        return location, wait
                    else:
                        location = self.drop_item(world_state)

            elif obj == "dish" and player_idx == 0:  # agent_index = 0
                location = self.drop_item(world_state)
            else:
                logger.info(p0_obj, action, obj)
                ValueError()

        elif action == "deliver":
            if p0_obj != "soup":
                location = self.mdp.get_empty_counter_locations(world_state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            logger.info(p0_obj, action, obj)
            ValueError()

        return location, wait

    def _shift_same_goal_pos(self, new_positions, change_idx):

        pos = new_positions[change_idx][0]
        ori = new_positions[change_idx][1]
        new_pos = pos
        new_ori = ori
        if (self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])[0] !=
                pos):
            new_pos, new_ori = self.mdp._move_if_direction(
                pos, ori, Action.ALL_ACTIONS[0])
        elif (self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])[0] !=
              pos):
            new_pos, new_ori = self.mdp._move_if_direction(
                pos, ori, Action.ALL_ACTIONS[1])
        elif (self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])[0] !=
              pos):
            new_pos, new_ori = self.mdp._move_if_direction(
                pos, ori, Action.ALL_ACTIONS[2])
        elif (self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])[0] !=
              pos):
            new_pos, new_ori = self.mdp._move_if_direction(
                pos, ori, Action.ALL_ACTIONS[3])
        else:
            logger.info("pos = ", pos)
            ValueError()

        new_positions[change_idx] = (new_pos, new_ori)

        return new_positions[0], new_positions[1]

    def mdp_action_state_to_world_state(self,
                                        action_idx,
                                        ori_state_idx,
                                        ori_world_state,
                                        with_argmin=False):
        new_world_state = ori_world_state.deepcopy()
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict,
                                                    ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]
        action = self.get_key_from_value(self.action_idx_dict, action_idx)

        possible_agent_motion_goals, ai_wait = self.map_action_to_location(
            ori_world_state,
            ori_mdp_state_key,
            self.action_dict[action][0],
            self.action_dict[action][1],
            player_idx=0,
        )
        possible_human_motion_goals, human_wait = self.map_action_to_location(
            ori_world_state,
            ori_mdp_state_key,
            self.action_dict[mdp_state_obj[-1]][0],
            self.action_dict[mdp_state_obj[-1]][1],
            p0_obj=mdp_state_obj[-2],
            player_idx=1,
        )  # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
        # get next position for AI agent
        agent_cost, agent_feature_pos = self.mp.min_cost_to_feature(
            ori_world_state.players[0].pos_and_or,
            possible_agent_motion_goals,
            with_motion_goal=True,
        )  # select the feature position that is closest to current player's position in world state
        new_agent_pos = (agent_feature_pos if agent_feature_pos is not None else
                         new_world_state.players[0].get_pos_and_or())
        human_cost, human_feature_pos = self.mp.min_cost_to_feature(
            ori_world_state.players[1].pos_and_or,
            possible_human_motion_goals,
            with_motion_goal=True,
        )
        new_human_pos = (human_feature_pos if human_feature_pos is not None else
                         new_world_state.players[1].get_pos_and_or())
        # logger.info(new_agent_pos, new_human_pos)

        if new_agent_pos == new_human_pos:
            new_agent_pos, new_human_pos = self._shift_same_goal_pos(
                [new_agent_pos, new_human_pos],
                np.argmax(np.array([agent_cost, human_cost])),
            )
            # logger.info("after shift =", new_agent_pos, new_human_pos)

        # update next position for AI agent
        if new_world_state.players[0].has_object():
            new_world_state.players[0].remove_object()
        if mdp_state_obj[0] != "None" and mdp_state_obj[0] != "soup":
            new_world_state.players[0].held_object = ObjectState(
                mdp_state_obj[0], new_agent_pos)
        new_world_state.players[0].update_pos_and_or(new_agent_pos[0],
                                                     new_agent_pos[1])

        # update next position for human
        if new_world_state.players[1].has_object():
            new_world_state.players[1].remove_object()
        if mdp_state_obj[-2] != "None" and mdp_state_obj[-2] != "soup":
            new_world_state.players[1].held_object = ObjectState(
                mdp_state_obj[-2], new_human_pos)
        new_world_state.players[1].update_pos_and_or(new_human_pos[0],
                                                     new_human_pos[1])

        total_cost = max([agent_cost, human_cost])
        if (
                ai_wait or human_wait
        ):  # if wait, then cost is sum of current tasks cost and one player's next task cost (est. as half map area length)
            total_cost = (agent_cost + human_cost + ((self.mdp.width - 1) +
                                                     (self.mdp.height - 1)) / 2)

        if with_argmin:
            return new_world_state, total_cost, [new_agent_pos, new_human_pos]

        return new_world_state, total_cost

    def world_to_state_keys(self, world_state, player, soup_finish,
                            other_player, belief):
        mdp_state_keys = []
        for i, b in enumerate(belief):
            mdp_state_key = self.world_state_to_mdp_state_key(
                world_state,
                player,
                other_player,
                self.get_key_from_value(self.subtask_idx_dict, i),
            )
            if self.get_mdp_state_idx(mdp_state_key) is not None:
                mdp_state_keys.append(
                    self.world_state_to_mdp_state_key(
                        world_state,
                        player,
                        other_player,
                        self.get_key_from_value(self.subtask_idx_dict, i),
                    ))
        return mdp_state_keys

    def joint_action_cost(self, world_state, goal_pos_and_or):
        (
            joint_action_plan,
            end_motion_state,
            plan_costs,
        ) = self.jmp.get_low_level_action_plan(world_state.players_pos_and_or,
                                               goal_pos_and_or,
                                               merge_one=True)
        # (
        #     joint_action_plan,
        #     end_state,
        #     plan_costs,
        # ) = self.mlp.get_embedded_low_level_action_plan(world_state,
        #                                                 goal_pos_and_or,
        #                                                 other_agent,
        #                                                 other_agent_idx)
        # logger.info(
        #     "joint_action_plan =",
        #     joint_action_plan,
        #     "; plan_costs =",
        #     plan_costs,
        # )
        if len(joint_action_plan) == 0:
            return (Action.INTERACT, None), 0
        return joint_action_plan[0], max(plan_costs)

    def step(
        self,
        world_state,
        mdp_state_keys,
        belief,
        agent_idx,
        low_level_action=False,
    ):
        """
        Compute plan cost that starts from the next qmdp state defined as
        next_state_v(). Compute the action cost of excuting a step towards the
        next qmdp state based on the current low level state information.
        """
        start_time = time.time()
        next_state_v = np.zeros((len(belief), len(self.action_dict)),
                                dtype=float)
        action_cost = np.zeros((len(belief), len(self.action_dict)),
                               dtype=float)
        qmdp_q = np.zeros((len(self.action_dict), len(belief)), dtype=float)
        # ml_action_to_low_action = np.zeros()

        # for each subtask, obtain next mdp state but with low level location
        # based on finishing excuting current action and subtask
        nxt_possible_mdp_state = []
        nxt_possible_world_state = []
        ml_action_to_low_action = []
        for i, mdp_state_key in enumerate(mdp_state_keys):
            mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
            if mdp_state_idx is not None:
                agent_action_idx_arr, next_mdp_state_idx_arr = np.where(
                    self.transition_matrix[:, mdp_state_idx] > 0.000001
                )  # returns array(action idx), array(next_state_idx)
                nxt_possible_mdp_state.append(
                    [agent_action_idx_arr, next_mdp_state_idx_arr])
                for j, action_idx in enumerate(agent_action_idx_arr):
                    next_state_idx = next_mdp_state_idx_arr[j]
                    (
                        after_action_world_state,
                        cost,
                        goals_pos,
                    ) = self.mdp_action_state_to_world_state(action_idx,
                                                             mdp_state_idx,
                                                             world_state,
                                                             with_argmin=True)
                    value_cost = self.compute_V(
                        after_action_world_state,
                        self.get_key_from_value(self.state_idx_dict,
                                                next_state_idx),
                        search_depth=100,
                    )
                    joint_action, one_step_cost = self.joint_action_cost(
                        world_state,
                        after_action_world_state.players_pos_and_or)

                    if not low_level_action:
                        next_state_v[i, action_idx] += (
                            value_cost *
                            self.transition_matrix[action_idx, mdp_state_idx,
                                                   next_state_idx])
                        # logger.info(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, action_idx] -= (
                            max(one_step_cost) *
                            self.transition_matrix[action_idx, mdp_state_idx,
                                                   next_state_idx])
                    else:
                        next_state_v[i, Action.ACTION_TO_INDEX[
                            joint_action[agent_idx]]] += (
                                value_cost *
                                self.transition_matrix[action_idx,
                                                       mdp_state_idx,
                                                       next_state_idx])
                        # logger.info(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, Action.ACTION_TO_INDEX[
                            joint_action[agent_idx]]] -= (
                                (one_step_cost) *
                                self.transition_matrix[action_idx,
                                                       mdp_state_idx,
                                                       next_state_idx])
                    # logger.info(
                    #     "action_idx =",
                    #     self.get_key_from_value(self.action_idx_dict,
                    #                             action_idx),
                    #     "; mdp_state_key =",
                    #     mdp_state_key,
                    #     "; next_state_key =",
                    #     self.get_key_from_value(self.state_idx_dict,
                    #                             next_state_idx),
                    # )
                    # logger.info("next_state_v =", next_state_v[i])
                    # logger.info("action_cost =", action_cost[i])

        q = self.compute_Q(belief, next_state_v, action_cost)
        # logger.info(q)
        action_idx = self.get_best_action(q)
        # logger.info(
        #     "get_best_action =",
        #     action_idx,
        #     "=",
        #     self.get_key_from_value(self.action_idx_dict, action_idx),
        # )
        # logger.info(f"It took {time.time() - start_time} seconds for this step")
        if low_level_action:
            return Action.INDEX_TO_ACTION[action_idx], None, low_level_action
        return (
            action_idx,
            self.action_dict[self.get_key_from_value(self.action_idx_dict,
                                                     action_idx)],
        )

    def belief_update(
        self,
        world_state,
        agent_player,
        soup_finish,
        human_player,
        belief_vector,
        prev_dist_to_feature,
        greedy=False,
    ):
        """
        Update belief based on both human player's game logic and also it's
        current position and action.
        """
        start_time = time.time()

        distance_trans_belief = np.zeros(
            (len(belief_vector), len(belief_vector)), dtype=float)
        human_pos_and_or = world_state.players[1].pos_and_or
        agent_pos_and_or = world_state.players[0].pos_and_or

        subtask_key = np.array([
            self.get_key_from_value(self.subtask_idx_dict, i)
            for i in range(len(belief_vector))
        ])

        # get next position for human
        human_obj = (human_player.held_object.name
                     if human_player.held_object is not None else "None")
        game_logic_prob = np.zeros((len(belief_vector)), dtype=float)
        dist_belief_prob = np.zeros((len(belief_vector)), dtype=float)
        for i, belief in enumerate(belief_vector):
            ## estimating next subtask based on game logic
            game_logic_prob[i] = (self._is_valid_object_subtask_pair(
                human_obj, subtask_key[i], soup_finish, greedy=greedy) * 1.0)

            ## tune subtask estimation based on current human's position and
            ## action (use minimum distance between features)
            possible_motion_goals, _ = self.map_action_to_location(
                world_state,
                None,
                self.subtask_dict[subtask_key[i]][0],
                self.subtask_dict[subtask_key[i]][1],
                p0_obj=human_obj,
                player_idx=1,
            )
            # get next world state from human subtask info (aka. mdp action
            # translate into medium level goal position)
            human_dist_cost, feature_pos = self.mp.min_cost_to_feature(
                human_pos_and_or, possible_motion_goals, with_argmin=True
            )  # select the feature position that is closest to current player's position in world state
            if str(feature_pos) not in prev_dist_to_feature:
                prev_dist_to_feature[str(feature_pos)] = human_dist_cost

            dist_belief_prob[i] = (self.mdp.height + self.mdp.width) + (
                prev_dist_to_feature[str(feature_pos)] - human_dist_cost)
            # dist_belief_prob[i] = ((self.mdp.height + self.mdp.width) -
            #                        human_dist_cost if human_dist_cost < np.inf
            #                        else (self.mdp.height + self.mdp.width))

            # update distance to feature
            prev_dist_to_feature[str(feature_pos)] = human_dist_cost

        game_logic_prob /= game_logic_prob.sum()
        dist_belief_prob /= dist_belief_prob.sum()

        game_logic_prob[game_logic_prob == 0.0] = 0.000001
        dist_belief_prob[dist_belief_prob == 0.0] = 0.000001

        new_belief = belief * game_logic_prob
        new_belief = new_belief * 0.7 * dist_belief_prob * 0.3

        new_belief /= new_belief.sum()
        # logger.info(
        #     f"It took {time.time() - start_time} seconds for belief update")

        return new_belief, prev_dist_to_feature

    def compute_V(self, next_world_state, mdp_state_key, search_depth=100):
        next_world_state_str = str(next_world_state)
        if next_world_state_str not in self.world_state_cost_dict:

            delivery_horizon = 2
            debug = False
            h_fn = Heuristic(self.mp).simple_heuristic
            start_world_state = next_world_state.deepcopy()
            if start_world_state.order_list is None:
                start_world_state.order_list = ["any"] * delivery_horizon
            else:
                start_world_state.order_list = \
                    start_world_state.order_list[:delivery_horizon]

            expand_fn = lambda state, ori_state_key: self.get_successor_states(
                state, ori_state_key)
            goal_fn = (lambda ori_state_key: len(self.state_dict[ori_state_key][
                2:]) <= 2)
            heuristic_fn = lambda state: h_fn(state)

            search_problem = SearchTree(start_world_state,
                                        goal_fn,
                                        expand_fn,
                                        heuristic_fn,
                                        max_iter_count=1e6,
                                        debug=debug)
            (
                path_end_state,
                cost,
                over_limit,
            ) = search_problem.bounded_A_star_graph_search(
                qmdp_root=mdp_state_key, info=False, cost_limit=search_depth)

            if over_limit:
                cost = self.optimal_plan_cost(path_end_state, cost)

            self.world_state_cost_dict[next_world_state_str] = cost

        # logger.info("self.world_state_cost_dict length =",
        #             len(self.world_state_cost_dict))
        return (self.mdp.height * self.mdp.width
               ) * 2 - self.world_state_cost_dict[next_world_state_str]

    def optimal_plan_cost(self, start_world_state, start_cost):
        self.world_state_to_mdp_state_key(
            start_world_state,
            start_world_state.players[0],
            start_world_state.players[1],
            subtask,  # TODO: subtask not defined
        )

    def compute_Q(self, b, v, c):
        # logger.info("b =", b)
        # logger.info("v =", v)
        # logger.info("c =", c)
        return b @ (v + c)

    @staticmethod
    def get_best_action(q):
        return np.argmax(q)

    def init_mdp(self):
        self.init_actions()
        self.init_human_aware_states(order_list=self.mdp.start_order_list)
        self.init_transition()

    def compute_mdp(self, filename):
        # start_time = time.time()

        final_filepath = os.path.join(PLANNERS_DIR, filename)
        self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)
        # logger.info("Total states =", self.num_states, "; Total actions =",
        #             self.num_actions)

        # logger.info(f"It took {time.time() - start_time} seconds to create "
        #             f"HumanSubtaskQMDPPlanner")
        # self.save_to_file(final_filepath)
        # tmp = input()
        # self.save_to_file(output_mdp_path)
        return
