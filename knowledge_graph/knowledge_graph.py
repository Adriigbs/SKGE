import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def create_nodes(self, entities):
        self.graph.add_node_from(entities)
    
    def create_node(self, node):
        self.graph.add_node(node)

    def remove_node(self, node):
        self.graph.remove_node(node)

    def add_edge(self, source, target, relation):
        """Adds edge between two nodes in the knowledge graph

        Args:
            source: Source node name of the directed edge
            target: Target node name of the directed edge
            relation: Relation name
        """
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            raise ValueError(f"One or both nodes {source} and {target} do not exist in the graph")
        self.graph.add_edge(source, target, key=relation)
    
    def remove_edge(self, source, target, relation):
        """Removes edge between two nodes in the knowledge graph

        Args:
            source: Source node name of the directed edge
            target: Target node name of the directed edge
            relation: Relation name
        """
        if not self.graph.has_edge(source, target, key=relation):
            raise ValueError(f"No edge exists between {source} and {target}")