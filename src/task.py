import logging
import os
import random

from src.mcts import MCTS_search


class MCTSTask:
    def __init__(
        self,
        time_limit=None,  # Time limit in milliseconds
        iteration_limit=None,  # Maximum number of iterations
        exploration_constant=1.0,  # UCT exploration constant
        roll_policy="greedy",  # 'greedy' or 'random'
        dataset=None,  # test dataset
        low=0,  # Minimum value
        high=1,  # Maximum value
        action_space=[], #这里写动作空间
        alpha=0.1,  # Value update rate
        value = 0, #初始值
    ):
        # Search parameters
        self.rank = int(os.environ.get("RANK", 0))
        self.action_space = action_space
        self.value = value

        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_policy = roll_policy

        # State
        self.dataset = dataset
        self.node_count = 0
        self.limit_type = time_limit
        self.root_node = None

        # Value range
        self.low_value = low
        self.high_value = high
        self.alpha = alpha

    def set_limit(self):
        """Set and validate the search limit type (time or iterations).

        Raises:
            ValueError: If both time and iteration limits are set, or if neither is set,
                      or if iteration limit is less than 1.
        """
        if self.time_limit is not None and self.iteration_limit is not None:
            raise ValueError("Cannot have both a time limit and an iteration limit")

        if self.time_limit is None and self.iteration_limit is None:
            raise ValueError("Must have either a time limit or an iteration limit")

        if self.time_limit is not None:
            self.limit_type = "time"
        else:
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = "iterations"


    def get_step_cost(self, current_node):
        """
        #这里写你的cost函数, 每一步的代价
        """
        return random.random()
    
    def get_reward(self, current_node):
        """
        #这里写你的reward函数, 最后是否完成任务
        """
        return random.random()

    def run(self):
        """
        Run MCTS search.

        Returns:
            TreeNode: Root node of the search tree
        """
        try:
            root_node, search_metric = MCTS_search(self)
            self.root_node = root_node  # Store for class-level access if needed
            return root_node
        except Exception as e:
            logging.error(f"Error during MCTS search: {str(e)}")
            raise
