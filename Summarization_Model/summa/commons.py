from .graph import Graph


def build_graph(sequence):
    graph = Graph()
    for i, item in enumerate(sequence):
        if not graph.has_node(item):
            graph.add_node(item, attrs=i)
    return graph


def remove_unreachable_nodes(graph):
    for node in graph.nodes():
        if sum(graph.edge_weight((node, other)) for other in graph.neighbors(node)) == 0:
            graph.del_node(node)
