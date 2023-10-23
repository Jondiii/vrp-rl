import numpy as np
from grafo import Grafo
from datetime import date
import matplotlib
import matplotlib.pyplot as plt
import os

"""
Esta clase almacena información de todas las rutas de un mismo problema.
Las rutas se almacenan de manera individual dentro de esta misma clase en formato de grafos, una clase
también implementada en este proyecto.
"""
class Rutas:
    """
    Recibe número de nodos del problema, número de vehículos y los valores máximos de estos. Es importante distinguir entre
    el número máximo de nodos y el número actual de nodos, ya que esto afecta a la reusabilidad del agente, lo mismo con los vehículos.
    Para representar las rutas, se usarán los valores actuales de estos elementos, no los máximos.
    """
    def __init__(self, nVehiculos, nNodos, maxNumVehiculos, maxNumNodos, demands, coordenadas, speeds, drawDemand = True):
        self.grafos = [] # Lista que contendrá las rutas individuales del problema, en forma de grafos.
        matplotlib.use('Agg') # Descomentar si se está trabajando en el server. Como esta clase se usa para visualizar rutas, si no se pone esto al hacer pruebas en el servidor, este peta.
        
        self.nNodos = nNodos
        self.nVehiculos = nVehiculos
        self.maxNumVehiculos = maxNumVehiculos
        self.maxNumNodos = maxNumNodos
        
        # Nos interesan solo los valores de demandas, coordenadas y velocidad correspondientes a los nodos actuales, no los máximos.
        self.demands = demands[:nNodos]
        self.coordenadas = coordenadas[:nNodos]
        self.speeds = speeds[:nVehiculos]

        # Si se debe escribir la demanda de cada nodo en el grafo o no.
        self.drawDemand = drawDemand
        self.crearGrafos()

    # Por cada vehículo, creamos un grafo.
    def crearGrafos(self):
        for i in range(self.nVehiculos):
            self.grafos.append(Grafo(self.nNodos, self.demands, self.coordenadas, self.speeds[i], self.drawDemand))

    # Método que calcula la distancia entre 2 nodos.
    def getDistance(self, vehiculo, nodo1, nodo2):
        return self.grafos[vehiculo].getDistance(nodo1, nodo2)

    # Método que calcula el tiempo que tarda un vehículo en ir de nodo 1 a nodo 2.
    def getTime(self, vehiculo, nodo1, nodo2):
        return self.grafos[vehiculo].getTime(self.getDistance(vehiculo, nodo1, nodo2))

    # Hace que un vehículo vaya desde nodo 1 a nodo 2 y marque a nodo 2 como visitado.
    def visitEdge(self, vehiculo, nodo1, nodo2):
        distancia, tiempo = self.grafos[vehiculo].visitEdge(nodo1, nodo2)
        return distancia, tiempo

    # Guarda una representación visual de los grafos (rutas) obtenidos.
    def guardarGrafos(self, fecha, directorio = 'grafos', name = 'fig', extension = '.png'):
        # Se guardan por fechas
        if fecha is None:
            fecha = str(date.today())
    
        directorio = os.path.join(directorio, fecha)

        if not os.path.exists(directorio):
            os.makedirs(directorio)

        # Se define cómo organizar los grafos. Máximo habrá 3 columnas, y tantas filas como sea necesario.
        num_columns = min(len(self.grafos), 3)
        num_rows = np.ceil(len(self.grafos) / num_columns).astype(int)

        # Se crea la figura y se asignan sus dimensiones.
        plt.clf()
        plt.figure(figsize=(5 * num_columns, 5 * num_rows))

        # En cada sección de la figura se crea un subplot, que contendrá un único grafo.
        for n, idGrafo in enumerate(range(len(self.grafos))):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            self.grafos[idGrafo].dibujarGrafo(ax = ax)

        # Este trozo de código añade números incrementales en función del número de imágenes que ya se hayan guardado en el propio directorio.
        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreFigura = os.path.join(directorio, name + '_' + siguiente_numero + extension)

        # Guardamos el plot y lo cerramos
        plt.savefig(nombreFigura)
        plt.close()
    
    # Método que hace lo mismo que el anterior, solo que en vez de dibujar cada ruta (grafo) por separado,
    # se dibujan todas a la vez, en un único plot. Suele quedar un poco caótico.
    def guardarGrafosSinglePlot(self, fecha, directorio = 'grafos', name = 'fig', extension = '.png'):
        if fecha is None:
            fecha = str(date.today())
    
        directorio = os.path.join(directorio, fecha)

        if not os.path.exists(directorio):
            os.makedirs(directorio)

        # Realmente tener tantas rutas pintadas a la vez es bastante confuso, por lo que no me he molestado en poner más de 10 colores.
        colorList = [
            "tab:red",
            "tab:orange",
            "tab:olive",
            "tab:green",
            "tab:cyan",
            "tab:blue",
            "tab:purple",
            "tab:pink",
            "tab:brown",
            "tab:gray"
        ]  # Para más info sobre colores: https://matplotlib.org/stable/tutorials/colors/colors.html

        plt.clf()
        ax = plt.subplot(1, 1, 1)
        
        # Pintamos cada ruta con un color distinto, sobre el mismo plot.
        for idGrafo, color in zip(range(len(self.grafos)), colorList):
            self.grafos[idGrafo].dibujarGrafo(ax = ax, edgeColor = color)

        # Este trozo de código añade números incrementales en función del número de imágenes que ya se hayan guardado en el propio directorio.
        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreFigura = os.path.join(directorio, name + '_' + siguiente_numero + extension)

        plt.savefig(nombreFigura)

        plt.close()

    # TODO / WIP. La idea de esto es que una ventana vaya mostrando las rutas según estas se van creando. Es decir, que primero se muestre
    # el depot, luego en otro frame se pinte el primer nodo a visitar, luego el segundo, etc.
    def getRutasVisual(self):
        num_columns = min(len(self.grafos), 3)
        num_rows = np.ceil(len(self.grafos) / num_columns).astype(int)

        plt.clf()

        for n, idGrafo in enumerate(range(len(self.grafos))):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            self.grafos[idGrafo].dibujarGrafo(ax = ax)

        plt.show()

