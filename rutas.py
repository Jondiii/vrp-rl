import numpy as np
from grafo import Grafo
from datetime import date
import matplotlib
import matplotlib.pyplot as plt
import os

class Rutas:
    def __init__(self, nVehiculos, nNodos, maxNumVehiculos, maxNumNodos, demands, coordenadas, speeds, drawDemand = True):
        self.grafos = []
        matplotlib.use('Agg') # Descomentar si se está trabajando en el server
        self.nNodos = nNodos
        self.nVehiculos = nVehiculos
        self.maxNumVehiculos = maxNumVehiculos
        self.maxNumNodos = maxNumNodos
        self.demands = demands[:nNodos]
        self.coordenadas = coordenadas[:nNodos]
        self.speeds = speeds[:nVehiculos]
        self.drawDemand = drawDemand
        self.crearGrafos()


    def crearGrafos(self):

        for i in range(self.nVehiculos):
            self.grafos.append(Grafo(self.nNodos, self.demands, self.coordenadas, self.speeds[i], self.drawDemand))


    def getDistance(self, vehiculo, nodo1, nodo2):
        return self.grafos[vehiculo].getDistance(nodo1, nodo2)


    def getTime(self, vehiculo, nodo1, nodo2):
        return self.grafos[vehiculo].getTime(self.getDistance(vehiculo, nodo1, nodo2))


    def visitEdge(self, vehiculo, nodo1, nodo2):
        distancia, tiempo = self.grafos[vehiculo].visitEdge(nodo1, nodo2)
        return distancia, tiempo


    def guardarGrafos(self, fecha, directorio = 'grafos', name = 'fig', extension = '.png'):
        if fecha is None:
            fecha = str(date.today())
    
        directorio = os.path.join(directorio, fecha)

        if not os.path.exists(directorio):
            os.makedirs(directorio)

        num_columns = min(len(self.grafos), 3)
        num_rows = np.ceil(len(self.grafos) / num_columns).astype(int)

        plt.clf()
        plt.figure(figsize=(5 * num_columns, 5 * num_rows))

        for n, idGrafo in enumerate(range(len(self.grafos))):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            self.grafos[idGrafo].dibujarGrafo(ax = ax)

        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreFigura = os.path.join(directorio, name + '_' + siguiente_numero + extension)

        plt.savefig(nombreFigura)

        plt.close()
    

    def guardarGrafosSinglePlot(self, fecha, directorio = 'grafos', name = 'fig', extension = '.png'):
        if fecha is None:
            fecha = str(date.today())
    
        directorio = os.path.join(directorio, fecha)

        if not os.path.exists(directorio):
            os.makedirs(directorio)

        # Para más info sobre colores: https://matplotlib.org/stable/tutorials/colors/colors.html
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
        ]

        plt.clf()
        ax = plt.subplot(1, 1, 1)
        
        for idGrafo, color in zip(range(len(self.grafos)), colorList):
            self.grafos[idGrafo].dibujarGrafo(ax = ax, edgeColor = color)

        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreFigura = os.path.join(directorio, name + '_' + siguiente_numero + extension)

        plt.savefig(nombreFigura)

        plt.close()


    def getRutasVisual(self):
        num_columns = min(len(self.grafos), 3)
        num_rows = np.ceil(len(self.grafos) / num_columns).astype(int)

        plt.clf()

        for n, idGrafo in enumerate(range(len(self.grafos))):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            self.grafos[idGrafo].dibujarGrafo(ax = ax)

        plt.show()

