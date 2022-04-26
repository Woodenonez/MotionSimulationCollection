class Tree:
    def __init__(self, root_node) -> None:
        self.node_list       = [root_node]
        self.parent_idx_list = [0]
        self.level_list      = [0]

    def add_node(self, node, parent_index:int):
        assert(0<=parent_index<len(self.node_list)),(f'Parent index exceeds limits 0~{len(self.node_list)-1}.')
        self.node_list.append(node)
        self.parent_idx_list.append(parent_index)
        self.level_list.append(len(self.return_node2root(len(self.node_list)-1))-1)

    def return_node2root(self, node_index:int) -> list:
        assert(0<=node_index<len(self.node_list)),(f'Node index exceeds limits 0~{len(self.node_list)-1}.')
        node2root_list = [self.node_list[node_index]]
        parent_idx = self.parent_idx_list[node_index]
        while parent_idx != 0:
            this_node = self.node_list[parent_idx]
            node2root_list.append(this_node)
            parent_idx = self.parent_idx_list[parent_idx]
        node2root_list.append(self.node_list[0])
        return node2root_list

    def print_tree_raw(self) -> None:
        print('Nodes:  ', self.node_list)
        print('Parents:', self.parent_idx_list)
        print('Level:  ', self.level_list)

    def print_tree_inlevel(self) -> None:
        print('#'*8, 'PRTIN TREE', '#'*8)
        print(f'root0: {self.node_list[0]}')
        for level in range(1, max(self.level_list)+1):
            node_index_in_level = [i for i,x in enumerate(self.level_list) if x==level]
            node_index_in_level = [x for x in node_index_in_level if x!=0]
            nodes_in_level = [x for i,x in enumerate(self.node_list) if i in node_index_in_level]
            print(f'node{level}:', nodes_in_level)
        print('#'*28)
