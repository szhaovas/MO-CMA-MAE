import itertools
import random

import numpy as np
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action


class GreedyHumanModel(Agent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do
    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS, for example
    forced_coordination.layout, in which an individual agent cannot complete the
    task on their own.
    """

    def __init__(
        self,
        mlp,
        hl_boltzmann_rational=False,
        ll_boltzmann_rational=False,
        hl_temp=1,
        ll_temp=1,
        auto_unstuck=True,
    ):
        self.mlp = mlp
        self.mdp = self.mlp.mdp

        # Bool for perfect rationality vs Boltzmann rationality for high level
        # and low level action selection
        self.hl_boltzmann_rational = (
            hl_boltzmann_rational  # For choices among high level goals of same type
        )
        self.ll_boltzmann_rational = (
            ll_boltzmann_rational  # For choices about low level motion
        )

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = hl_temp
        self.ll_temperature = ll_temp

        # Whether to automatically take an action to get the agent unstuck if
        # it's in the same state as the previous turn. If false, the agent is
        # history-less, while if true it has history.
        self.auto_unstuck = auto_unstuck

        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None

    def actions(self, states, agent_indices):
        actions_and_infos_n = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions_and_infos_n.append(self.action(state))
        return actions_and_infos_n

    def action(self, state, eps=0):
        possible_motion_goals = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
            start_pos_and_or, possible_motion_goals)

        if self.ll_boltzmann_rational and chosen_goal[0] == start_pos_and_or[0]:
            chosen_action, action_probs = self.boltzmann_rational_ll_action(
                start_pos_and_or, chosen_goal)

        if self.auto_unstuck:
            # HACK: if two agents get stuck, select an action at random that
            # would change the player positions if the other player were not to
            # move
            if (self.prev_state is not None and state.players_pos_and_or
                    == self.prev_state.players_pos_and_or):
                if self.agent_index == 0:
                    joint_actions = list(
                        itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
                elif self.agent_index == 1:
                    joint_actions = list(
                        itertools.product([Action.STAY], Action.ALL_ACTIONS))
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    new_state, _, _, _ = self.mlp.mdp.get_state_transition(
                        state, j_a)
                    if new_state.player_positions != self.prev_state.player_positions:
                        unblocking_joint_actions.append(j_a)

                if len(unblocking_joint_actions) > 0:
                    chosen_action = unblocking_joint_actions[np.random.choice(
                        len(unblocking_joint_actions))][self.agent_index]
                else:
                    chosen_action = Action.STAY
                action_probs = self.a_probs_from_action(chosen_action)

                state.players[self.agent_index].stuck_log += [1]
            else:
                state.players[self.agent_index].stuck_log += [0]

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state

        # eps-greedy
        if random.random() < eps:
            chosen_action = (Action.ALL_ACTIONS[np.random.randint(6)], {})[0]

        if chosen_action == Action.STAY:
            state.players[self.agent_index].active_log += [0]
        else:
            state.players[self.agent_index].active_log += [1]

        return chosen_action, {"action_probs": action_probs}

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """
        For each motion goal, consider the optimal motion plan that reaches the
        desired location. Based on the plan's cost, the method chooses a motion
        goal (either boltzmann rationally or rationally), and returns the plan
        and the corresponding first action on that plan.
        """
        if self.hl_boltzmann_rational:
            possible_plans = [
                self.mlp.mp.get_plan(start_pos_and_or, goal)
                for goal in motion_goals
            ]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx, action_probs = self.get_boltzmann_rational_action_idx(
                plan_costs, self.hl_temperature)
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
        else:
            chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(
                start_pos_and_or, motion_goals)
            action_probs = self.a_probs_from_action(chosen_goal_action)
        return chosen_goal, chosen_goal_action, action_probs

    @staticmethod
    def get_boltzmann_rational_action_idx(costs, temperature):
        """Chooses index based on softmax probabilities obtained from cost
        array."""
        costs = np.array(costs)
        softmax_probs = np.exp(-costs * temperature) / np.sum(
            np.exp(-costs * temperature))
        action_idx = np.random.choice(len(costs), p=softmax_probs)
        return action_idx, softmax_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlp.mp.get_plan(
                start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def boltzmann_rational_ll_action(self,
                                     start_pos_and_or,
                                     goal,
                                     inverted_costs=False):
        """
        Computes the plan cost to reach the goal after taking each possible low
        level action. Selects a low level action boltzmann rationally based on
        the one-step-ahead plan costs. If `inverted_costs` is True, it will make
        a boltzmann "irrational" choice, exponentially favouring high cost plans
        rather than low cost ones.
        """
        future_costs = []
        for action in Action.ALL_ACTIONS:
            pos, orient = start_pos_and_or
            new_pos_and_or = self.mdp._move_if_direction(pos, orient, action)
            _, _, plan_cost = self.mlp.mp.get_plan(new_pos_and_or, goal)
            sign = (-1)**int(inverted_costs)
            future_costs.append(sign * plan_cost)

        action_idx, action_probs = self.get_boltzmann_rational_action_idx(
            future_costs, self.ll_temperature)
        return Action.ALL_ACTIONS[action_idx], action_probs

    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]
        In this method, X (e.g. deliver the soup, pick up an onion, etc) is
        chosen based on a simple set of greedy heuristics based on the current
        state.
        Effectively, will return a list of all possible locations Y in which the
        selected medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager

        counter_objects = self.mlp.mdp.get_counter_objects_dict(
            state, list(self.mlp.mdp.terrain_pos_dict["X"]))
        pot_states_dict = self.mlp.mdp.get_pot_states(state)

        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if not player.has_object():

            if curr_order == "any":
                ready_soups = (pot_states_dict["onion"]["ready"] +
                               pot_states_dict["tomato"]["ready"])
                cooking_soups = (pot_states_dict["onion"]["cooking"] +
                                 pot_states_dict["tomato"]["cooking"])
            else:
                ready_soups = pot_states_dict[curr_order]["ready"]
                cooking_soups = pot_states_dict[curr_order]["cooking"]

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = (other_player.has_object() and
                              other_player.get_object().name == "dish")

            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                next_order = None
                if state.num_orders_remaining > 1:
                    next_order = state.next_order

                if next_order == "onion":
                    motion_goals = am.pickup_onion_actions(counter_objects)
                elif next_order == "tomato":
                    motion_goals = am.pickup_tomato_actions(counter_objects)
                elif next_order is None or next_order == "any":
                    motion_goals = am.pickup_onion_actions(
                        counter_objects) + am.pickup_tomato_actions(
                            counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == "onion":
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == "tomato":
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)

            elif player_obj.name == "dish":
                motion_goals = am.pickup_soup_with_dish_actions(
                    pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == "soup":
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [
            mg for mg in motion_goals
            if self.mlp.mp.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg)
        ]

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [
                mg for mg in motion_goals
                if self.mlp.mp.is_valid_motion_start_goal_pair(
                    player.pos_and_or, mg)
            ]
            assert len(motion_goals) != 0

        return motion_goals


class MediumQMdpPlanningAgent(Agent):

    def __init__(
        self,
        mdp_planner,
        greedy=False,
        other_agent=None,
        delivery_horizon=1,
        logging_level=0,
        auto_unstuck=False,
    ):
        # self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mdp_planner = mdp_planner
        self.logging_level = logging_level
        self.auto_unstuck = auto_unstuck
        self.other_agent = other_agent
        self.greedy_known = greedy
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None
        self.prev_dist_to_feature = {}
        self.belief = np.full(
            (len(self.mdp_planner.subtask_dict)),
            1.0 / len(self.mdp_planner.subtask_dict),
            dtype=float,
        )

    def mdp_action_to_low_level_action(self, state, state_strs,
                                       action_object_pair):
        # map back the medium level action to low level action
        ai_agent_obj = (state.players[0].held_object.name
                        if state.players[0].held_object is not None else "None")
        # logger.info(ai_agent_obj)
        possible_motion_goals, wait = self.mdp_planner.map_action_to_location(
            state,
            state_strs[0],
            action_object_pair[0],
            action_object_pair[1],
            p0_obj=ai_agent_obj,
            player_idx=0,
        )

        # initialize
        action = Action.STAY
        minimum_cost = 100000.0
        # logger.info(state)
        # logger.info("possible_motion_goals =", possible_motion_goals)
        if not wait:
            for possible_location in possible_motion_goals:
                motion_goal_locations = self.mdp_planner.mp.motion_goals_for_pos[
                    possible_location]
                for motion_goal_location in motion_goal_locations:
                    if self.mdp_planner.mp.is_valid_motion_start_goal_pair(
                        (state.players[0].position,
                         state.players[0].orientation),
                            motion_goal_location,
                    ):
                        action_plan, _, cost = self.mdp_planner.mp._compute_plan(
                            (state.players[0].position,
                             state.players[0].orientation),
                            motion_goal_location,
                        )
                        if cost < minimum_cost:
                            minimum_cost = cost
                            action = action_plan[0]
        return action

    def action(self, state):
        low_level_action = False
        num_item_in_pot = 0
        pot_pos = []
        if state.objects is not None and len(state.objects) > 0:
            for obj_pos, obj_state in state.objects.items():
                if obj_state.name == "soup" and obj_state.state[
                        1] > num_item_in_pot:
                    num_item_in_pot = obj_state.state[1]
                    pot_pos = obj_pos

        self.belief, self.prev_dist_to_feature = self.mdp_planner.belief_update(
            state,
            state.players[0],
            num_item_in_pot,
            state.players[1],
            self.belief,
            self.prev_dist_to_feature,
            greedy=self.greedy_known,
        )
        mdp_state_keys = self.mdp_planner.world_to_state_keys(
            state, state.players[0], num_item_in_pot, state.players[1],
            self.belief)
        action, action_object_pair, low_level_action = self.mdp_planner.step(
            state,
            mdp_state_keys,
            self.belief,
            self.agent_index,
            low_level_action=True)

        if not low_level_action:
            action = self.mdp_action_to_low_level_action(
                state, mdp_state_keys, action_object_pair)

        # logger.info("action =", action, "; action_object_pair =",
        #             action_object_pair)
        action_probs = self.a_probs_from_action(action)
        if self.auto_unstuck:
            action, action_probs = self.resolve_stuck(state, action,
                                                      action_probs)
            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state

        if action == Action.STAY:
            state.players[self.agent_index].active_log += [0]
        else:
            state.players[self.agent_index].active_log += [1]
        # logger.info("\nState =", state)
        # logger.info("Subtasks:", self.mdp_planner.subtask_dict.keys())
        # logger.info("Belief =", self.belief)
        # logger.info(
        #     "Max belief =",
        #     list(self.mdp_planner.subtask_dict.keys())[np.argmax(self.belief)],
        # )
        # logger.info("Action =", action, "\n")

        return action, {"action_probs": action_probs}

    def resolve_stuck(self, state, chosen_action, action_probs):
        # HACK: if two agents get stuck, select an action at random that would
        # change the player positions if the other player were to move
        if (self.prev_state is not None and
                state.players_pos_and_or == self.prev_state.players_pos_and_or):
            joint_actions = list(
                itertools.product(Action.MOTION_ACTIONS, Action.MOTION_ACTIONS))
            unblocking_joint_actions = []
            for j_a in joint_actions:
                new_state, _, _, _ = self.mdp_planner.mdp.get_state_transition(
                    state, j_a)
                if new_state.player_positions != self.prev_state.player_positions:
                    unblocking_joint_actions.append(j_a)

            if len(unblocking_joint_actions) > 0:
                chosen_action = unblocking_joint_actions[np.random.choice(
                    len(unblocking_joint_actions))][self.agent_index]
            else:
                chosen_action = Action.STAY
            action_probs = self.a_probs_from_action(chosen_action)

            state.players[self.agent_index].stuck_log += [1]

        else:
            state.players[self.agent_index].stuck_log += [0]

        return chosen_action, action_probs
