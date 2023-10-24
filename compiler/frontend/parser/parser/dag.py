from compiler.frontend.parser.node.node import Node

class DAG:
    def __init__(self, nodes : list[Node]):
        self._nodes : list[Node] = []
        for node in nodes:
            self.append_node(node)

    def __str__(self):
        result : str = "DAG" + "\n"*2
        result += [str(node)+"\n"*2 for node in self._nodes]
        return result

    def append_node(self, node : Node):
        name = node.get_name()
        if self._is_name_in_list(name):
            raise Exception("Error: node name is unique in dag")
        self._nodes.append(node)

    def get_node(self, name : str):
        index : int = self._get_node_index_by_name(name)
        if index == -1:
            raise Exception("Error: node not found in dag")
        return self._nodes[index]

    def remove_node(self, name : str):
        index : int = self._get_node_index_by_name(name)
        if index == -1:
            raise Exception("Error: node not found in dag")
        self._nodes.pop(index)

    def get_list_names(self) -> list[str]:
        names : list[str] = []
        for node in self._nodes:
            name = node.get_name()
            names.append(name)
        return names

    def check_if_dag() -> bool:
        pass

    def traversal_dag_and_generate_code() -> str:
        pass

    def _is_name_in_list(self, name : str) -> bool:
        return name in self.get_list_names()

    def _get_node_index_by_name(self, name : str) -> int:
        i : int = 0
        for node in self._nodes:
            if node.get_name() == name:
                return i
        return -1