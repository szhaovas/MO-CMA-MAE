import logging
import time

from overcooked_ai_py.planning.search import PriorityQueue, SearchNode

logger = logging.getLogger(__name__)


class SearchTree:
    """
    A class to help perform tree searches of various types. Once a goal state is
    found, returns a list of tuples containing (action, state) pairs. This
    enables to recover the optimal action and state path.

    Args:
        root (state): Initial state in our search
        goal_fn (func): Takes in a state and returns whether it is a goal state
        expand_fn (func): Takes in a state and returns a list of (action,
            successor, action_cost) tuples
        heuristic_fn (func): Takes in a state and returns a heuristic value
    """

    def __init__(self,
                 root,
                 goal_fn,
                 expand_fn,
                 heuristic_fn,
                 max_iter_count=10e8,
                 debug=False):
        self.debug = debug
        self.root = root
        self.is_goal = goal_fn
        self.expand = expand_fn
        self.heuristic_fn = heuristic_fn
        self.max_iter_count = max_iter_count

    def A_star_graph_search(self, info=False):
        """
        Performs a A* Graph Search to find a path to a goal state
        """
        if info:
            logger.info("A_star_graph_search")
        start_time = time.time()
        iter_count = 0
        seen = set()
        pq = PriorityQueue()

        root_node = SearchNode(self.root,
                               action=None,
                               parent=None,
                               action_cost=0,
                               debug=self.debug)
        pq.push(root_node, self.estimated_total_cost(root_node))
        while not pq.isEmpty():
            curr_node = pq.pop()
            iter_count += 1

            if self.debug and iter_count % 1000 == 0:
                logger.info([p[0] for p in curr_node.get_path()])
                logger.info(iter_count)

            curr_state = curr_node.state

            if curr_state in seen:
                continue

            seen.add(curr_state)
            if iter_count > self.max_iter_count:
                logger.info(
                    "Expanded more than the maximum number of allowed states")
                raise TimeoutError("Too many states expanded expanded")

            if self.is_goal(curr_state):
                elapsed_time = time.time() - start_time
                if info:
                    logger.info(
                        f"Found goal after: \t{elapsed_time:.2f} seconds,   "
                        f"\t{iter_count} state expanded "
                        f"({len(seen) / iter_count:.2f} unique) \t "
                        f"~{iter_count / elapsed_time:.2f} expansions/s")
                return curr_node.get_path(), curr_node.backwards_cost

            successors = self.expand(curr_state)

            for action, child, cost in successors:
                child_node = SearchNode(child,
                                        action,
                                        parent=curr_node,
                                        action_cost=cost,
                                        debug=self.debug)
                pq.push(child_node, self.estimated_total_cost(child_node))

        logger.info(
            f"Path for last node expanded: "
            f"{[p[0] for p in curr_node.get_path()]}"
        )
        logger.info(f"State of last node expanded: {curr_node.state}")
        logger.info(
            f"Successors for last node expanded: {self.expand(curr_state)}")
        raise TimeoutError("A* graph search was unable to find any goal state.")

    def bounded_A_star_graph_search(self,
                                    qmdp_root=None,
                                    info=False,
                                    cost_limit=10e8):
        """
        Performs a A* Graph Search to find a path to a goal state
        """
        if info:
            logger.info("A_star_graph_search")
        start_time = time.time()
        iter_count = 0
        seen = set()
        pq = PriorityQueue()

        root_node = SearchNode(self.root,
                               action=qmdp_root,
                               parent=None,
                               action_cost=0,
                               debug=self.debug)
        pq.push(root_node, self.estimated_total_cost(root_node))
        while not pq.isEmpty():
            curr_node = pq.pop()
            iter_count += 1

            if self.debug and iter_count % 1000 == 0:
                logger.info([p[0] for p in curr_node.get_path()])
                logger.info(iter_count)

            curr_state = curr_node.state
            curr_qmdp_state = curr_node.action

            if curr_qmdp_state in seen:
                continue

            seen.add(curr_qmdp_state)
            if iter_count > self.max_iter_count:
                logger.info(
                    "Expanded more than the maximum number of allowed states")
                raise TimeoutError("Too many states expanded expanded")

            if self.is_goal(curr_qmdp_state):
                elapsed_time = time.time() - start_time
                if info:
                    logger.info(
                        f"Found goal after: \t{elapsed_time:.2f} seconds,   "
                        f"\t{iter_count} state expanded "
                        f"({len(seen) / iter_count:.2f} unique) \t "
                        f"~{iter_count / elapsed_time:.2f} expansions/s")
                return curr_node.state, curr_node.backwards_cost, False

            successors = self.expand(curr_state, curr_qmdp_state)

            for qmdp_state, child, cost in successors:
                child_node = SearchNode(
                    child,
                    qmdp_state,
                    parent=curr_node,
                    action_cost=cost,
                    debug=self.debug,
                )
                est_total_cost = self.estimated_total_cost(child_node)
                pq.push(child_node, self.estimated_total_cost(child_node))

        logger.info(
            f"Path for last node expanded: "
            f"{[p[0] for p in curr_node.get_path()]}"
        )
        logger.info(f"State of last node expanded: {curr_node.state}")
        logger.info(
            f"Successors for last node expanded: "
            f"{self.expand(curr_state, curr_qmdp_state)}"
        )
        raise TimeoutError("A* graph search was unable to find any goal state.")

    def estimated_total_cost(self, node):
        """
        Calculates the estimated total cost of going from node to goal

        Args:
            node (SearchNode): node of the state we are interested in

        Returns:
            float: h(s) + g(s), where g is the total backwards cost
        """
        return node.backwards_cost + self.heuristic_fn(node.state)
