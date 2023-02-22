import networkx as nx
import numpy as np

class Grafo:
    def __init__(self, nNodos, demands, coordenadas, nDepots = 1, drawDemand = True) -> None:
        self.nNodos = nNodos
        self.nDepots = nDepots
        self.demands = demands
        self.drawDemand = drawDemand

        self.graph = nx.complete_graph(nNodos)
        self.coordenadas = {   # Enumerate pone números delante de los elementos de una lista, así que la i
                            # se queda con el número que haya puesto el enumerate.
            i: coordinates for i, coordinates in enumerate(coordenadas)
        }
        
        nx.set_node_attributes(self.graph, self.coordenadas, "coordinates")

        depotList = np.zeros(nNodos)
        depotList[0] = 1 # Marcamos como depot el primer nodo de la lista 
        depotDict = {i: depot for i, depot in enumerate(depotList)}
        nx.set_node_attributes(self.graph, depotDict, "depot")
        
        self.demands[0] = 0 # El depot no tiene demanda
        node_demand = {i: d for i, d in enumerate(self.demands)}
        nx.set_node_attributes(self.graph, node_demand, "demand")

        # Definimos atributos del grafo
        nx.set_edge_attributes(self.graph, False, "visited")
        nx.set_node_attributes(self.graph, "black", "node_color")
        
        self.graph.nodes[0]["node_color"] = "red"

        # Offset a usar a la hora de dibujar labels
        self.offset = np.array([0, 0.045])

    def visitEdge(self, sourceNode, targetNode):
        if sourceNode == targetNode:
            return
        
        self.graph.edges[sourceNode, targetNode]["visited"] = True

    def dibujarGrafo(self, ax):
        posicion = nx.get_node_attributes(self.graph, "coordinates")
        coloresNodos = nx.get_node_attributes(self.graph, "node_color").values()

        # Primero dibujamos los nodos
        nx.draw_networkx_nodes(self.graph, posicion, node_color=coloresNodos, ax=ax, node_size=100)

        # Después dibujamos las aristas
        aristas = [x for x in self.graph.edges(data = True) if x[2]["visited"]]
        nx.draw_networkx_edges(
            self.graph,
            posicion,
            alpha=0.5,
            edgelist=aristas,
            edge_color="red",
            ax=ax,
            width=1.5,
        )

        if self.drawDemand:
            posicionLabelDemanda = {k: (v + self.offset) for k, v in posicion.items()}
            labelDemanda = nx.get_node_attributes(self.graph, "demand")
            labelDemanda = {k: np.round(v, 2) for k, v in labelDemanda.items()}
            nx.draw_networkx_labels(
                self.graph, posicionLabelDemanda, labels=labelDemanda, ax=ax
            )

    def getDistance(self, node1, node2):
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        return np.linalg.norm(self.graph.nodes[node1]["coordinates"] - self.graph.nodes[node2]["coordinates"])