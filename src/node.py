class TreeNode:
    def __init__(self, action, parent=None):
        self.action = action  # Current action/step text
        self.state = None  # Full state including parent actions
        self.parent = parent  # Parent node
        self.visit_count = 0  # Number of times node was visited
        self.value = 0  # Node value/score
        self.children = {}  # Child nodes mapping action to node
        self.depth = 0 if parent is None else parent.depth + 1  # Depth in tree
        self.isFullyExpanded = False  # Whether all children are expanded
        self.visit_order = 0  # Order in which node was visited
        self.isTerminal = False  # Whether the node has a solution

    def append_children(self, action):
        node = TreeNode(action, self)
        node.update_state_from_parent()
        self.children.update({action: node})
        return self

    def update_state_from_parent(self):
        #这里继承父节点的state，总体来说是一个state seq，你要自己定义state
        #这个函数我的state是以字符串举例的，你可能需要用列表存
        if self.parent is None:
            self.state = None #(初始状态，你自己写)
        else:
            self.state = self.parent.state + self.transition(self.action) #（状态转移）

    def transition(self, action):
        #这里写你的transition函数
        #机器人做什么动作，返回next state
        pass

    def update_value(self, value):
        self.value = value
