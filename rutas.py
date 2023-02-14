from typing import List

from grafo import Grafo
import matplotlib.pyplot as plt
import numpy as np

class Rutas:
    def __init__(self, nGrafos, nNodos, nDepots, drawDemand) -> None:
        
        assert (
            nNodos >= nDepots
        ), "Number of nodes should be lower than number of depots"

        self.nGrafos = nGrafos
        self.nNodos = nNodos
        self.nDepots = nDepots
        self.drawDemand = drawDemand
        self.grafos: List[Grafo] = []

        for _ in range(nGrafos):
            self.grafos.append(Grafo(nNodos, nDepots, plot_demand=drawDemand))


    def getDistance(self, graph_idx: int, node_idx_1: int, node_idx_2: int) -> float:
        """
        Calculates the euclid distance between the two nodes 
        within a single graph in the VRPNetwork.

        Args:
            graph_idx (int): Index of the graph
            node_idx_1 (int): Source node
            node_idx_2 (int): Target node

        Returns:
            float: Euclid distance between the two nodes
        """
        return self.grafos[graph_idx].euclid_distance(node_idx_1, node_idx_2)


    def getDistances(self, paths) -> np.ndarray:
        """
        Calculatest the euclid distance between
        each node pair in paths.

        Args:
            paths (nd.array): Shape num_graphs x 2
                where the second dimension denotes
                [source_node, target_node].

        Returns:
            np.ndarray: Euclid distance between each
                node pair. Shape (num_graphs,) 
        """
        return np.array(
            [
                self.getDistance(index, source, dest)
                for index, (source, dest) in enumerate(paths)
            ]
        )

    def getDepots(self) -> np.ndarray:
        """
        Get the depots of every graph within the network.

        Returns:
            np.ndarray: Returns nd.array of shape
                (num_graphs, num_depots).
        """

        depos_idx = np.zeros((self.nGrafos, self.nDepots), dtype=int)

        for i in range(self.nGrafos):
            depos_idx[i] = self.grafos[i].depots

        return depos_idx

    
    def getDemands(self) -> np.ndarray:
        """
        Returns the demands for each node in each graph.

        Returns:
            np.ndarray: Demands of each node in shape 
                (num_graphs, num_nodes, 1)
        """
        demands = np.zeros(shape=(self.nGrafos, self.nDepots, 1))
        for i in range(self.nGrafos):
            demands[i] = self.grafos[i].demand

        return demands

    def draw(self, graph_idxs: np.ndarray) -> None:
        """
        Draw multiple graphs in a matplotlib grid.

        Args:
            graph_idxs (np.ndarray): Idxs of graphs which get drawn.
                Expected to be of shape (x, ). 
        
        Returns:
            np.ndarray: Plot as rgb-array of shape (width, height, 3).
        """

        num_columns = min(len(graph_idxs), 3)
        num_rows = np.ceil(len(graph_idxs) / num_columns).astype(int)

        # plot each graph in a 3 x num_rows grid
        plt.clf()
        fig = plt.figure(figsize=(5 * num_columns, 5 * num_rows))

        for n, graph_idx in enumerate(graph_idxs):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            self.graphs[graph_idx].draw(ax=ax)

        #plt.show()

        # convert to plot to rgb-array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return image

    def visit_edges(self, transition_matrix: np.ndarray) -> None:
        """
        Visits each edges specified in the transition matrix.

        Args:
            transition_matrix (np.ndarray): Shape num_graphs x 2
                where each row is [source_node_idx, target_node_idx].
        """
        for i, row in enumerate(transition_matrix):
            self.grafos[i].visit_edge(row[0], row[1])


    def getGraphPositions(self) -> np.ndarray:
        """
        Returns the coordinates of each node in every graph as
        an ndarray of shape (num_graphs, num_nodes, 2) sorted
        by the graph and node index.

        Returns:
            np.ndarray: Node coordinates of each graph. Shape
                (num_graphs, num_nodes, 2)
        """

        node_positions = np.zeros(shape=(len(self.grafos), self.num_nodes, 2))
        for i, graph in enumerate(self.grafos):
            node_positions[i] = graph.node_positions

        return node_positions