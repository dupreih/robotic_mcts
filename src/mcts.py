import math
import random
import time

import numpy

from src.node import TreeNode


def MCTS_search(mcts_task):
    """
    Performs Monte Carlo Tree Search within specified time or iteration limits.
    Returns the root node, solution node (if found), and search duration/iterations.
    """
    # Validate and set search limits
    mcts_task.set_limit()

    root_node = TreeNode("")
    search_start_time = time.time()

    if mcts_task.limit_type == "time":
        search_end_time = search_start_time + mcts_task.time_limit / 1000
        while time.time() < search_end_time:
            print(
                f"<begin a new search round, elapsed time: {time.time() - search_start_time:.2f}s>\n"
            )
            root_node = executeRound(root_node, mcts_task)
        search_metric = time.time() - search_start_time

    else:
        for iteration_count in range(mcts_task.iteration_limit):
            print(
                f"<Begin search round {iteration_count + 1}/{mcts_task.iteration_limit}>\n"
            )
            root_node = executeRound(root_node, mcts_task)
        search_metric = mcts_task.iteration_limit

    return root_node, search_metric


def executeRound(root_node, mcts_task):
    # Execute a selection-expansion-simulation-backpropagation round
    print("-" * 30, "phase selection", "-" * 30, "\n")
    selected_node = selectNode(root_node, mcts_task)
    print(f"Selected node: {selected_node.state}, depth: {selected_node.depth}\n")

    print("-" * 30, "phase expansion", "-" * 30, "\n")
    if selected_node.isTerminal:
        print("This is a terminal node, no further expansion required.\n")
    else:
        selected_node = expand(selected_node, mcts_task)
        print(
            f"Complete expansion!, expanded node count: {len(selected_node.children)}\n"
        )

    print("-" * 30, "phase simulation", "-" * 30, "\n")
    if selected_node.isTerminal:
        print("This is a terminal node, no further simulation required.\n")
    else:
        rollout_node = getBestChild(selected_node, mcts_task)
        terminal_node, value = greedyPolicy(rollout_node, mcts_task)
        # Update value with exponential moving average
        rollout_node.value = (
            rollout_node.value * (1 - mcts_task.alpha) + value * mcts_task.alpha
        )
        rollout_node.visit_count += 1

    print("-" * 30, "phase backpropagation", "-" * 30, "\n")
    back_propagate(terminal_node, value)

    return root_node


def selectNode(current_node, mcts_task):
    while current_node.isFullyExpanded:
        current_node = getBestChild(current_node, mcts_task)
    if current_node.isTerminal:
        return current_node
    return current_node


def getBestChild(parent_node, mcts_task):
    best_value = mcts_task.low_value
    best_child_nodes = []
    for child_node in parent_node.children.values():
        # UCB1 formula for node selection
        if child_node.visit_count > 0:
            exploitation_term = child_node.value
            exploration_term = mcts_task.exploration_constant * math.sqrt(
                2 * math.log(parent_node.visit_count) / child_node.visit_count
            )
            ucb_value = exploitation_term + exploration_term
        else:
            ucb_value = child_node.value + 1.0  # 确保未访问的节点会被选中

        if ucb_value > best_value:
            best_value = ucb_value
            best_child_nodes = [child_node]
        elif ucb_value == best_value:
            best_child_nodes.append(child_node)
    return random.choice(best_child_nodes)


def expand(current_node, mcts_task):
    
    for action in mcts_task.action_space:
        if action not in list(current_node.children.keys()):
            current_node.append_children(action)
            child_node = current_node.children[action]
            child_node.update_value(mcts_task.get_step_value(child_node))
            child_node.visit_order = mcts_task.node_count
            mcts_task.node_count += 1

    current_node.isFullyExpanded = True
    return current_node


def greedyPolicy(current_node, mcts_task):

    while not current_node.isTerminal:
        current_node.visit_count += 1
        mcts_task.value += mcts_task.get_step_cost(current_node)
        current_node = getBestChild(current_node, mcts_task)

    mcts_task.value += mcts_task.get_reward(current_node)

    terminal_node = current_node

    return terminal_node, mcts_task.value

def back_propagate(terminal_node, value):
    current_node = terminal_node
    while current_node is not None:
        current_node.visit_count += 1
        if current_node.isFullyExpanded:
            child_weighted_values = [
                child.value * child.visit_count
                for child in current_node.children.values()
            ]
            total_child_visits = sum(
                child.visit_count for child in current_node.children.values()
            )
            if total_child_visits > 0:
                current_node.value = sum(child_weighted_values) / total_child_visits
        elif current_node.isTerminal:
            current_node.value = value
        current_node = current_node.parent
