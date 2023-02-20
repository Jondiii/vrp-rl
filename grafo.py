import networkx as nx
import numpy as np

class Grafo:
    def __init__(self, nNodos, nDepots, drawDemand) -> None:
        
        self.nNodos = nNodos
        self.nDepots = nDepots
        self.drawDemand = drawDemand

        coordenadas = np.array([[43.326651753558124, -3.0324346743529342],
                                [43.31934608579387, -3.021276684889593],
                                [43.30865194199736, -3.0062113198058227],
                                [43.304779525190796, -3.03779701338381],
                                [43.30665330597163, -3.0129919443897286]])

        
        self.graph = nx.complete_graph(nNodos)
        nodos = {   # Enumerate pone números delante de los elementos de una lista, así que la i
                            # se queda con el número que haya puesto el enumerate.
            i: coordinates for i, coordinates in enumerate(coordenadas)
        }
        nx.set_node_attributes(self.graph, nodos, "coordinates")


        # De momento seleccionamos un depot al azar.
        self.depots = np.random.choice(nNodos, size=nDepots, replace=False)
        one_hot = np.zeros(nNodos)
        one_hot[self.depots] = 1
        one_hot_dict = {i: depot for i, depot in enumerate(one_hot)}
        nx.set_node_attributes(self.graph, one_hot_dict, "depot")


        # Seleccionamos las demandas, de nuevo aleatorias.
        C = 0.2449 * nNodos + 26.12  # linear reg on values from paper
        demand = np.random.uniform(low=1, high=10, size=(nNodos, 1)) / C
        demand[self.depots] = 0
        node_demand = {i: d for i, d in enumerate(demand)}
        nx.set_node_attributes(self.graph, node_demand, "demand")

        # Definimos atributos del grafo
        nx.set_edge_attributes(self.graph, False, "visited")
        nx.set_node_attributes(self.graph, "black", "node_color")
        
        for node in self.depots:
            self.graph.nodes[node]["node_color"] = "red"
