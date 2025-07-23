from matplotlib import pyplot as plt
import daft
from daft import PGM
import pgmpy
from pgmpy.inference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork


def convert_pgm_to_pgmpy(pgm):
    """Takes a Daft PGM object and converts it to a pgmpy BayesianModel"""
    edges = [(edge.node1.name, edge.node2.name) for edge in pgm._edges]
    model = DiscreteBayesianNetwork(edges)
    return model

def get_pgm(number):
    if number==1:
        pgm = PGM(shape=[4, 4])

        pgm.add_node(daft.Node('X', r"X", 1, 1))
        pgm.add_node(daft.Node('Y', r"Y", 3, 1))
        pgm.add_node(daft.Node('A', r"A", 1, 3))
        pgm.add_node(daft.Node('B', r"B", 2, 3))
        pgm.add_node(daft.Node('C', r"C", 3, 3))
        pgm.add_node(daft.Node('D', r"D", 2, 2))
        pgm.add_node(daft.Node('E', r"E", 2, 1))

        pgm.add_edge('X', 'E')
        pgm.add_edge('A', 'X')
        pgm.add_edge('A', 'B')
        pgm.add_edge('B', 'C')
        pgm.add_edge('D', 'B')
        pgm.add_edge('D', 'E')
        pgm.add_edge('E', 'Y')

        pgm.render()
    if number==2:
        pgm = PGM(shape=[4, 4])

        pgm.add_node(daft.Node('X', r"X", 1, 1))
        pgm.add_node(daft.Node('Y', r"Y", 3, 1))
        pgm.add_node(daft.Node('A', r"A", 2, 1.75))
        pgm.add_node(daft.Node('B', r"B", 2, 3))

        pgm.add_edge('X', 'Y')
        pgm.add_edge('X', 'A')
        pgm.add_edge('B', 'A')
        pgm.add_edge('B', 'X')
        pgm.add_edge('B', 'Y')

        pgm.render()

    if number==3:
        pgm = PGM(shape=[4, 4])

        pgm.add_node(daft.Node('X', r"X", 1, 1))
        pgm.add_node(daft.Node('Y', r"Y", 3, 1))
        pgm.add_node(daft.Node('A', r"A", 1, 3))
        pgm.add_node(daft.Node('B', r"B", 2, 2))
        pgm.add_node(daft.Node('C', r"C", 3, 3))

        pgm.add_edge('A', 'X')
        pgm.add_edge('A', 'B')
        pgm.add_edge('C', 'B')
        pgm.add_edge('C', 'Y')

        pgm.render()

    if number==4:
        pgm = PGM(shape=[4, 4])

        pgm.add_node(daft.Node('X', r"X", 1, 1))
        pgm.add_node(daft.Node('Y', r"Y", 3, 1))
        pgm.add_node(daft.Node('A', r"A", 1, 3))
        pgm.add_node(daft.Node('B', r"B", 2, 2))
        pgm.add_node(daft.Node('C', r"C", 3, 3))

        pgm.add_edge('A', 'X')
        pgm.add_edge('A', 'B')
        pgm.add_edge('C', 'B')
        pgm.add_edge('C', 'Y')
        pgm.add_edge("X", "Y")
        pgm.add_edge("B", "X")

        pgm.render()

    if number==5:
        pgm = PGM(shape=[4, 4])

        pgm.add_node(daft.Node('X', r"X", 1, 1))
        pgm.add_node(daft.Node('Y', r"Y", 3, 1))
        pgm.add_node(daft.Node('A', r"A", 1, 3))
        pgm.add_node(daft.Node('B', r"B", 3, 3))
        pgm.add_node(daft.Node('C', r"C", 1, 2))
        pgm.add_node(daft.Node('D', r"D", 2, 2))
        pgm.add_node(daft.Node('E', r"E", 3, 2))
        pgm.add_node(daft.Node('F', r"F", 2, 1))

        pgm.add_edge('X', 'F')
        pgm.add_edge('F', 'Y')
        pgm.add_edge('C', 'X')
        pgm.add_edge('A', 'C')
        pgm.add_edge('A', 'D')
        pgm.add_edge('D', 'X')
        pgm.add_edge('D', 'Y')
        pgm.add_edge('B', 'D')
        pgm.add_edge('B', 'E')
        pgm.add_edge('E', 'Y')

        pgm.render()

    if number==6:
        pgm = PGM(shape=[4, 3])

        pgm.add_node(daft.Node('X', r"X", 1, 1))
        pgm.add_node(daft.Node('Y', r"Y", 3, 1))
        pgm.add_node(daft.Node('A', r"A", 2, 1))
        pgm.add_node(daft.Node('B', r"B", 2, 2))

        pgm.add_edge('X', 'A')
        pgm.add_edge('A', 'Y')
        pgm.add_edge('B', 'X')
        pgm.add_edge('B', 'Y')

        pgm.render()

    if number==7:
        pgm = PGM(shape=[4, 3])

        pgm.add_node(daft.Node("X", r"X", 1, 2))
        pgm.add_node(daft.Node("Y", r"Y", 3, 2))
        pgm.add_node(daft.Node("A", r"A", 2, 2))
        pgm.add_node(daft.Node("B", r"B", 2, 1))

        pgm.add_edge("X", "A")
        pgm.add_edge("A", "Y")
        pgm.add_edge("A", "B")

        pgm.render()

    plt.savefig("out/pgm"+str(number)+".png")
    return pgm


def backdoor_paths(number):
    pgm=get_pgm(number)
    graph = convert_pgm_to_pgmpy(pgm)
    inference = CausalInference(graph)



    """bg=inference.get_proper_backdoor_graph('X','Y')
    viz1 = bg.to_graphviz()
    viz1.draw('out/bg'+str(number)+'.png', prog='dot')"""

    print(f"Are there are active backdoor paths? {not inference.is_valid_backdoor_adjustment_set('X', 'Y')}")
    adj_sets = inference.get_all_backdoor_adjustment_sets("X", "Y")
    print(f"If so, what's the possible backdoor adjustment sets? {adj_sets}")
    fd_adj_sets = inference.get_all_frontdoor_adjustment_sets("X", "Y")
    print(f"What's the possible front adjustment sets? {fd_adj_sets}")