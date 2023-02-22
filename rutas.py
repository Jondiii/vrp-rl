import numpy as np
from grafo import Grafo
from datetime import date
import matplotlib.pyplot as plt
import os

class Rutas:
    def __init__(self, nVehiculos, nNodos, demands, coordenadas, drawDemand = True):
        self.grafos = []

        self.nNodos = nNodos
        self.nVehiculos = nVehiculos
        self.demands = demands
        self.coordenadas = coordenadas
        self.drawDemand = drawDemand

        self.crearGrafos()


    def crearGrafos(self):
        for i in range(self.nVehiculos):
            self.grafos.append(Grafo(self.nNodos, self.demands, self.coordenadas, self.drawDemand))


    def getDistance(self, vehiculo, nodo1, nodo2):
        return self.grafos[vehiculo].getDistance(nodo1, nodo2)


    def visitEdge(self, vehiculo, nodo1, nodo2):
        self.grafos[vehiculo].visitEdge(nodo1, nodo2)


    def guardarGrafos(self, directorio = 'grafos', name = 'fig', extension = '.png'):
        directorio = os.path.join(directorio, str(date.today()))

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
