"""Repair config and repair module"""

import numpy as np
from docplex.mp.model import Context, Model

from .level import INT_TO_OBJ_TYPES


class RepairModule:
    """
    Class for repairing Overcooked levels.
    Args:
        cost_type: "flow" to use flow-based cost, "hamming" for Hamming
            distance (default "flow")
        use_cont: True if continuous variables should be used whenever
            possible (default False)
        discard_suboptimal: True if suboptimal repairs should be discarded
            (default False)
        time_limit: Deterministic time limit for MIP in ticks (default 1500)
    """

    def __init__(
        self,
        cost_type: str,
        use_cont: bool,
        discard_suboptimal: bool,
        time_limit: int,
    ):
        self.cost_type = cost_type
        self.use_cont = use_cont
        self.discard_suboptimal = discard_suboptimal
        self.time_limit = time_limit

        self.cnt = 0

    @staticmethod
    def add_object_placement(mdl, all_objects):
        """
        Adds constraints that ensure exactly one object is present in each cell
        Args:
            mdl: the milp model
            all_objects: a list of all object variables [[W_i], [P_i], ...]
        """
        # Transpose the given matrix and ensure exactly one object per graph
        # node
        for cur_node in zip(*all_objects):
            mdl.add_constraint(sum(cur_node) == 1)

    def add_reachability(
        self, mdl, graph, source_objects, sink_objects, blocking_objects
    ):
        """
        Adds reachability constraints to milp
        Args:
            mdl: the milp model
            graph: an adjacency list
            source_objects: objects that must reach the sink objects
                [[P_i], ...]
            sink_objects: objects that must be reached by the source objects
                [[K_i], [D_i], ...]
            blocking_objects: a list of object types that impede movement
                [[W_i], ...]
        """
        var_type = mdl.continuous_var if self.use_cont else mdl.integer_var
        # Transpose the blocking objects matrix so all blocking objects for
        # a given node are easily accessible.
        blocking = list(zip(*blocking_objects))

        # Setup a flow network for each edge in the graph
        n_nodes = len(graph)
        # Add a flow variable for each edge in the graph
        # flow: the flow leaving node i
        # rev: flow edges entering node i
        flow = [[] for i in range(n_nodes)]
        rev = [[] for i in range(n_nodes)]
        for i, neighbors in enumerate(graph):
            for j in neighbors:
                f = var_type(
                    name="p_{}_{}-{}".format(i, j, self.cnt), lb=0, ub=n_nodes
                )
                flow[i].append(f)
                rev[j].append(f)

        # Add supply and demand variables for the source and sink
        supplies = []
        demands = []
        for i in range(n_nodes):
            f = var_type(name="p_s_{}-{}".format(i, self.cnt), lb=0, ub=n_nodes)
            supplies.append(f)
            f = var_type(name="p_{}_t-{}".format(i, self.cnt), lb=0, ub=1)
            demands.append(f)
        # Add a flow conservation constraint for each node (outflow == inflow)
        for i in range(n_nodes):
            mdl.add_constraint(
                supplies[i] + sum(rev[i]) == demands[i] + sum(flow[i])
            )

        # Add capacity constraints for each edge ensuring that no flow passes
        # through a blocking object
        for i, neighbors in enumerate(flow):
            blocking_limits = [n_nodes * b for b in blocking[i]]
            for f in neighbors:
                mdl.add_constraint(f + sum(blocking_limits) <= n_nodes)

        # Place a demand at this object location if it contains a sink type
        # object.
        sinks = list(zip(*sink_objects))
        for i in range(n_nodes):
            mdl.add_constraint(sum(sinks[i]) == demands[i])

        # Allow this node to have supply if it contains a source object
        sources = list(zip(*source_objects))
        for i in range(n_nodes):
            capacity = sum(n_nodes * x for x in sources[i])
            mdl.add_constraint(supplies[i] <= capacity)

    def add_edit_distance(self, mdl, graph, objects):
        """
        Adds edit distance cost function and constraints for fixing the level
        with minimal edits.
        Args:
            mdl: the milp model
            graph: an adjacency list denoting allowed movement
            objects: a list [([(T_i, O_i)], Cm, Cc), ...] representing the cost
                of moving each object by one edge (Cm) and the cost of an add or
                delete (Cc). T_i represents the object variable at node i O_i is
                a boolean value denoting whether node i originally contained
                T_i.
        """
        costs = []
        if self.cost_type == "hamming":
            for objects_in_graph, cost_move, cost_change in objects:
                for cur_var, did_contain in objects_in_graph:
                    if did_contain:
                        costs.append(cost_change * (1 - cur_var))
                    else:
                        costs.append(cost_change * cur_var)

        elif self.cost_type == "flow":
            var_type = mdl.continuous_var if self.use_cont else mdl.integer_var
            for obj_id, (objects_in_graph, cost_move, cost_change) in enumerate(
                objects
            ):

                # Setup a flow network for each edge in the graph
                n_nodes = len(graph)
                # Add a flow variable for each edge in the graph
                # flow: the flow leaving node i
                # rev: flow edges entering node i
                flow = [[] for i in range(n_nodes)]
                rev = [[] for i in range(n_nodes)]
                for i, neighbors in enumerate(graph):
                    for j in neighbors:
                        f = var_type(
                            name="edit({})_{}_{}".format(obj_id, i, j),
                            lb=0,
                            ub=n_nodes,
                        )
                        costs.append(cost_move * f)
                        flow[i].append(f)
                        rev[j].append(f)

                # Add a supply if the object was in the current location.
                # Demands go everywhere.
                demands = []
                waste = []
                num_supply = 0
                for i, (cur_var, did_contain) in enumerate(objects_in_graph):
                    f = var_type(
                        name="edit({})_{}_t".format(obj_id, i), lb=0, ub=1
                    )
                    demands.append(f)

                    # Add a second sink that eats any flow that doesn't find a
                    # home. The cost of this flow is deleting the object.
                    f = var_type(
                        name="edit({})_{}_t2".format(obj_id, i),
                        lb=0,
                        ub=n_nodes,
                    )
                    costs.append(cost_change * f)
                    waste.append(f)

                    # Flow conservation constraint (inflow == outflow)
                    if did_contain:
                        # If we had a piece of this type in the current node,
                        # match it to the outflow
                        mdl.add_constraint(
                            1 + sum(rev[i])
                            == demands[i] + sum(flow[i]) + waste[i]
                        )
                        num_supply += 1
                    else:
                        mdl.add_constraint(
                            sum(rev[i]) == demands[i] + sum(flow[i]) + waste[i]
                        )

                # Ensure we place a piece of this type to match it to the
                # demand.
                for (cur_var, did_contain), node_demand in zip(
                    objects_in_graph, demands
                ):
                    mdl.add_constraint(node_demand <= cur_var)

                # Ensure that the source and sink have the same flow.
                mdl.add_constraint(num_supply == sum(demands) + sum(waste))

        else:
            raise NotImplementedError(
                f"Cost type {self.cost_type} not implemented"
            )

        mdl.minimize(mdl.sum(costs))

    def add_reachability_helper(
        self, source_labels, sink_labels, blocking_labels, mdl, adj, objs
    ):
        source_objects = [
            objs[INT_TO_OBJ_TYPES.index(label)] for label in source_labels
        ]
        sink_objects = [
            objs[INT_TO_OBJ_TYPES.index(label)] for label in sink_labels
        ]
        blocking_objects = [
            objs[INT_TO_OBJ_TYPES.index(label)] for label in blocking_labels
        ]
        self.add_reachability(
            mdl, adj, source_objects, sink_objects, blocking_objects
        )

    def repair_lvl(self, np_lvl: np.ndarray, seed: int = None):
        """
        Repairs an Overcooked level
        Args:
            np_lvl: Unrepaired level with shape (lvl_height x lvl_width)
            seed: Random seed to use

        Returns:
            Repaired level
        """
        n, m = np_lvl.shape

        deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # Build an adjacency list for the dynamics of Overcooked
        n_nodes = n * m
        adj = [[] for i in range(n_nodes)]
        border_nodes = []
        for i in range(n_nodes):
            cur_row = i // m
            cur_col = i % m
            is_border = False
            for dr, dc in deltas:
                nxt_row = cur_row + dr
                nxt_col = cur_col + dc
                if 0 <= nxt_row < n and 0 <= nxt_col < m:
                    j = nxt_row * m + nxt_col
                    adj[i].append(j)
                else:
                    is_border = True
            if is_border:
                border_nodes.append(i)

        context = Context.make_default_context()
        context.cplex_parameters.threads = 1
        context.cplex_parameters.dettimelimit = self.time_limit
        if seed is not None:
            context.cplex_parameters.randomseed = seed

        with Model(context=context) as mdl:
            objs = []
            for obj_label in INT_TO_OBJ_TYPES:
                curr_type = [
                    mdl.integer_var(
                        name="obj_{}_{}".format(obj_label, i), lb=0, ub=1
                    )
                    for i in range(n_nodes)
                ]
                objs.append(curr_type)

            # ensure one cell contains one obj_type
            self.add_object_placement(mdl, objs)

            # Ensure that all cells on the boundary are walls
            not_allowed_on_border = []
            for label in "12 ":
                i = INT_TO_OBJ_TYPES.index(label)
                not_allowed_on_border += [objs[i][j] for j in border_nodes]
            mdl.add_constraint(sum(not_allowed_on_border) <= 0)

            # Player1 and 2 show up exactly once
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("1")]) == 1)
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("2")]) == 1)

            # At least one onion, dish plate, pot, and serve point
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("O")]) >= 1)
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("D")]) >= 1)
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("P")]) >= 1)
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("S")]) >= 1)

            # Upper bound number of onion, dish plate, pot, and serve point
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("O")]) <= 2)
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("D")]) <= 2)
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("P")]) <= 2)
            mdl.add_constraint(sum(objs[INT_TO_OBJ_TYPES.index("S")]) <= 2)

            # Upper bound total number of onion, dish plate, pot, and serve
            # point
            mdl.add_constraint(
                sum(objs[INT_TO_OBJ_TYPES.index("O")])
                + sum(objs[INT_TO_OBJ_TYPES.index("D")])
                + sum(objs[INT_TO_OBJ_TYPES.index("P")])
                + sum(objs[INT_TO_OBJ_TYPES.index("S")])
                <= 6
            )

            # reachability
            source_labels = "1"
            sink_labels = "ODPS2 "
            blocking_labels = "XODPS"
            self.add_reachability_helper(
                source_labels, sink_labels, blocking_labels, mdl, adj, objs
            )

            # add edit distance objective
            objects = []
            cost_move = 1
            cost_change = 20
            for cur_idx, cur_obj in enumerate(objs):
                objects_in_graph = []
                for r in range(n):
                    for c in range(m):
                        i = r * m + c
                        objects_in_graph.append(
                            (cur_obj[i], cur_idx == np_lvl[r, c])
                        )
                objects.append((objects_in_graph, cost_move, cost_change))

            self.add_edit_distance(mdl, adj, objects)

            solution = mdl.solve()

            if solution is None:
                return None

            if (
                self.discard_suboptimal
                and "optimal" not in solution.solve_details.status
            ):
                return None

            def get_idx_from_variables(solution, node_id):
                for i, obj_var in enumerate(objs):
                    if np.isclose(solution.get_value(obj_var[node_id]), 1):
                        return i
                return -1

            # Extract the new level from the milp model
            new_lvl = np.zeros((n, m))
            for r in range(n):
                for c in range(m):
                    i = r * m + c
                    new_lvl[r, c] = get_idx_from_variables(solution, i)

            return new_lvl.astype(np.uint8)
