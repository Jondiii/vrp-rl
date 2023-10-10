import networkx as nx
import numpy as np

# Esta clase es la encargada de representar rutas de manera individual.
class Grafo:
    """
    Recibe el número de nodos, sus demandas, las coordenadas y la velocidad del vehículo.
    """
    def __init__(self, nNodos, demands, coordenadas, speed, nDepots = 1, drawDemand = True) -> None:
        self.nNodos = nNodos
        self.nDepots = nDepots
        self.demands = demands
        self.speed = speed

        # Indica si se debe poner la demanda junto cada nodo o no.
        self.drawDemand = drawDemand

        # Creamos un grafo completo (es decir, un grafo donde todos los nodos están conectados entre sí)
        self.graph = nx.complete_graph(nNodos)
        self.coordenadas = {   # Enumerate pone números delante de los elementos de una lista, así que la i
                               # se queda con el número que haya puesto el enumerate.
            i: coordinates for i, coordinates in enumerate(coordenadas)
        }
        
        # Añadimos las coordenadas de cada nodo
        nx.set_node_attributes(self.graph, self.coordenadas, "coordinates")

        # Tenemos que especificar qué nodos tienen la propiedad de depot. Esto se hace creando una lista donde todos
        # los nodos tengan un 0 en depot, salvo el primer nodo, que sí será el depot y tendrá un 1.
        depotList = np.zeros(nNodos)
        depotList[0] = 1 # Marcamos como depot el primer nodo de la lista 
        depotDict = {i: depot for i, depot in enumerate(depotList)}
        nx.set_node_attributes(self.graph, depotDict, "depot")
        
        # Añadimos el atributo demandas. El depot no tiene demanda nunca.
        self.demands[0] = 0 # El depot no tiene demanda
        node_demand = {i: d for i, d in enumerate(self.demands)}
        nx.set_node_attributes(self.graph, node_demand, "demand")

        # Marcamos todos los nodos como no visitados
        nx.set_node_attributes(self.graph, False, "visited")

        # Pintamos los nodos de negro, salvo el depot, que será rojo.
        nx.set_node_attributes(self.graph, "black", "node_color")
        self.graph.nodes[0]["node_color"] = "red"

        # Definimos atributos del grafo. Marcamos los arcos como no visitados.
        nx.set_edge_attributes(self.graph, False, "visited")

        # Offset a usar a la hora de dibujar labels. Esto hace que se seleccione un valor aleatorio entre 0 y 0.045
        # con el que se desplazará ligeramente el lugar en el que se dibujan los labels, para que no queden justo
        # en medio del nodo.
        self.offset = np.array([0, 0.045])

    # Método que hace que un nodo sea visitado, devolviendo información sobre la distancia recorrida y el tiempo empleado.
    def visitEdge(self, sourceNode, targetNode):
        if sourceNode == targetNode:
            return 0, 0
        
        distancia = self.getDistance(sourceNode, targetNode)
        tiempo = self.getTime(distancia)

        self.graph.edges[sourceNode, targetNode]["visited"] = True
        self.graph.nodes[targetNode]["visited"] = True

        return distancia, tiempo

    # Método encargado de dibujar una ruta concreta.
    def dibujarGrafo(self, ax, edgeColor = "red"):

        # Método que comprueba si un nodo ha sido visitado o no.
        def isNodeVisited(node):
            return self.graph.nodes[node]["visited"] == True
        
        posicion = nx.get_node_attributes(self.graph, "coordinates")

        # Primero dibujamos los nodos en base al método creado. Solo se usarán los métodos que devuelvan True.
        subGrafo = nx.subgraph_view(self.graph, isNodeVisited)

        # Seleccionamos los colores de los nodos (negro)
        node_colors = [c for _, c in subGrafo.nodes(data="node_color")]

        # Pinta todos los nodos a la vez.
        nx.draw_networkx_nodes(
            self.graph,
            posicion,
            node_color = node_colors,
            ax=ax,
            nodelist = subGrafo.nodes(),
            node_size=100
            )

        # Después dibujamos las aristas/arcos que hayan sido visitados
        aristas = [x for x in self.graph.edges(data = True) if x[2]["visited"]]

        nx.draw_networkx_edges(
            self.graph,
            posicion,
            alpha=0.5,
            edgelist=aristas,
            edge_color=edgeColor,
            ax=ax,
            width=1.5,
        )

        # Ponemos el número de cada nodo
        posicionIDNodo = {k: (v) for k, v in posicion.items()}
        labelIDNodo = {id: id for id, _ in subGrafo.nodes.items()}
        nx.draw_networkx_labels(
            self.graph, posicionIDNodo, labels=labelIDNodo, ax=ax, font_color = "white", font_size = 8
        )

        # Pintamos la demanda
        if self.drawDemand:
            posicionLabelDemanda = {k: (v + self.offset) for k, v in posicion.items()}
            labelDemanda = nx.get_node_attributes(subGrafo, "demand")
            labelDemanda = {k: np.round(v, 2) for k, v in labelDemanda.items()}
            nx.draw_networkx_labels(
                self.graph, posicionLabelDemanda, labels=labelDemanda, ax=ax
            )



    def getDistance(self, node1, node2):
        # Supuestamente esta es la forma más realista de calcular la distancia entre dos coordenadas:
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        return np.linalg.norm(self.graph.nodes[node1]["coordinates"] - self.graph.nodes[node2]["coordinates"])

    # Calculamos el tiempo en horas que se tarda en recorrer la distancia.
    def getTime(self, distancia):
        return distancia * 60 / self.speed