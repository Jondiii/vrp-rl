import gym
from gym import spaces
import numpy as np
import pandas as pd
from rutas import Rutas
import copy
import os
from datetime import date

class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nVehiculos, nNodos, maxCapacity = 100, maxNodeCapacity = 6, speed = 70, twMin = None, twMax = None, seed = 6, multiTrip = False, singlePlot = False):        
        np.random.seed(seed)
        
        # Características del entorno
        self.multiTrip = multiTrip
        self.singlePlot = singlePlot
        self.nNodos = nNodos + 1 # +1 del depot
        self.nVehiculos = nVehiculos
        self.currTime = np.zeros(shape=(self.nVehiculos,1))

        # Características de los nodos
        self.n_coordenadas = np.random.rand(nNodos+1, 2) # [0, nNodos), por lo que hay que sumarle +1
        self.n_originalDemands = np.random.randint(low = 1, high = maxNodeCapacity, size=self.nNodos) * 5 # Demandas múltiplo de 5
        self.n_demands = copy.deepcopy(self.n_originalDemands)
        self.n_maxNodeCapacity = maxNodeCapacity

        # Características de los vehículos
        self.v_maxCapacity = maxCapacity
        self.v_speed = speed
        self.v_loads = np.zeros(shape=self.nVehiculos) + self.v_maxCapacity
        self.v_speeds = np.zeros(shape=self.nVehiculos) + self.v_speed

        # Cálculo de matrices de distancia
        self.createMatrixes()

        # Creamos las time windows
        self.crearTW(twMin, twMax)

        # Tantas acciones como (número de nodos + depot) * número de vehículos
        self.action_space = spaces.Discrete(self.nNodos * self.nVehiculos)

        # Aquí se define cómo serán las observaciones que se pasarán al agente.
        # Se usa multidiscrete para datos que vengan en formato array. Hay que definir el tamaño de estos arrays
        # y sus valores máximos. La primera, "visited", podrá tomar un máximo de 2 valores en cada posición del array
        # (1 visitado - 0 sin visitar), por lo que le pasamos un array [2,2,...,2]
        self.observation_space = spaces.Dict({
            "n_visited" :  spaces.MultiDiscrete(np.zeros(shape=self.nNodos) + 2),
            "v_curr_position" : spaces.MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.nNodos),
            "v_loads" : spaces.MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.v_maxCapacity + 1), # SOLO se pueden usar enteros
            "n_demands" : spaces.MultiDiscrete(np.zeros(shape=self.nNodos) + self.n_maxNodeCapacity * 5),
            "v_curr_time" : spaces.Box(low = 0, high = float('inf'), shape = (self.nVehiculos,), dtype=float),
            "n_distances" : spaces.Box(low = 0, high = float('inf'), shape = (self.nVehiculos * self.nNodos,), dtype=float),
            "n_timeLeftTWClose" : spaces.Box(low = 0, high = float('inf'), shape = (self.nVehiculos * self.nNodos,), dtype=float)
        })


    def step(self, action):
        # supongamos que nNodos = 6, nVehiculos = 2 y action = 6 * 2 + 2
        # Calculamos el vehículo que realiza la acción
        vehiculo = action // self.nNodos
        # vehiculo = 1, es decir, el segundo vehículo

        action = action % self.nNodos
        # action = 14   %  6 = 2 # Sería visitar el tercer nodo

        # Comprobar si la acción es válida
        if not self.checkAction(action, vehiculo):
            return self.getState(), -1, False, dict()
        
        # Eliminar el lugar que se acaba de visitar de las posibles acciones
        self.visited[action] = 1

        if self.multiTrip:
            if action == 0:
                self.v_loads[vehiculo] = self.v_maxCapacity
                self.visited[action] = 0 # Si lo que se ha visitado es el depot, no lo marcamos como visitado 

        self.v_loads[vehiculo] -= self.n_demands[action] # Añadimos la demanda del nodo a la carga del vehículo

        # Marcamos la visita en el grafo
        distancia, tiempo = self.rutas.visitEdge(vehiculo, self.v_posicionActual[vehiculo], action)

        # Ponemos la demanda del nodo a 0
        self.n_demands[action] = 0
        
        # Calcular la recompensa. Será inversamente proporcional a la distancia recorrida. 
        reward = self.getReward(distancia)

        # Actualizar posición del vehículo que realice la acción.
        self.v_posicionActual[vehiculo] = action

        # Añadir el nodo a la ruta del vehículo.
        self.v_ordenVisitas[vehiculo].append(action)

        # Actualizar las distancias a otros nodos
        self.n_distances[vehiculo] = self.distanceMatrix[action]

        # Actualizar tiempo de recorrido del vehículo que realice la acción
        self.currTime[vehiculo] += tiempo

        # Comprobar si se ha llegado al final del episodio
        done = self.isDone()

        return self.getState(), reward, done, dict()


    def reset(self):
        self.visited = np.zeros(shape=(self.nNodos))

        if self.multiTrip:
            self.visited[0] = 0
        else:
            self.visited[0] = 1 # El depot comienza como visitado

        self.v_posicionActual = np.zeros(shape = self.nVehiculos)
        self.v_loads = np.zeros(shape=self.nVehiculos,) + self.v_maxCapacity
        self.n_demands = copy.deepcopy(self.n_originalDemands)
        self.currTime = np.zeros(shape=(self.nVehiculos), dtype = float)

        self.n_distances = np.zeros(shape = (self.nVehiculos, self.nNodos))

        for i in range(0, self.nVehiculos):
            self.n_distances[i] = self.distanceMatrix[0]

        # Creamos un conjunto de rutas nuevo
        self.rutas = Rutas(self.nVehiculos, self.nNodos, self.n_demands, self.n_coordenadas, self.v_speeds)
        
        self.v_ordenVisitas = []

        for _ in range(self.nVehiculos):
            self.v_ordenVisitas.append([0])
        
        self.done = False

        return self.getState()


    def checkAction(self, action, vehiculo):
        if self.visited[action] == 1:
            return False
        
        if self.v_loads[vehiculo] - self.n_demands[action] < 0:
            return False


        if self.minTW[action] > self.currTime[vehiculo]:
            #print("MIN: {} - {}".format(self.minTW[action], self.currTime[vehiculo]))
            return False

        if self.maxTW[action] < self.currTime[vehiculo]:
            #print("MAX: {} - {}".format(self.maxTW[action], self.currTime[vehiculo]))
            return False

        return True
    

    def getState(self):
        obs = dict()
        obs["n_visited"] = self.visited
        obs["v_curr_position"] = self.v_posicionActual
        obs["v_loads"] = self.v_loads
        obs["n_demands"] = self.n_demands   # No sé si tiene mucho sentido pasarle la demanda cuando esta no va a cambiar...
                                            # A no ser que pongamos la demanda de un nodo a 0 cuando esta sea recogida.
        obs["v_curr_time"] = self.currTime
        obs["n_distances"] = self.n_distances.flatten()

        obs["n_timeLeftTWClose"] = self.getTimeLeftTWClose().flatten()

        return obs


    def isDone(self): # can't DO: cambiar esto de orden, primero comprobar vehículo y después nodos --> no se puede por lo de marcar el depot como no visitado
        if self.multiTrip:
            allVisited = np.all(self.visited[1:] == 1)

            if allVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.currTime)
                return True
            
        if np.all(self.v_posicionActual == 0):
            allVisited = np.all(self.visited == 1)
            if allVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.currTime)
                return True
            
        else:
            allVisited = np.all(self.visited == 1)
            if allVisited:
                self.visited[0] = 0 
        
        return False
    

    def getReward(self, distancia):
        if distancia == 0:
            return 0.0

        reward = round(1/abs(distancia), 2)

        return reward
        
    # Current time tiene shape (nVehiculos,), mientras que el repeat devuelve shape (nNodos, nvehiculos)
    # No se puede hacer broadcast de esos rangos, pero con np.array([self.currTime]).T tenemos que curr time tiene shape (nVehiculos,1)
    # y con eso sí se puede hacer la resta que queremos
    def getTimeLeftTWClose(self):
        twClose = np.repeat([self.maxTW], repeats=self.nVehiculos, axis=0) - np.array([self.currTime]).T

        return twClose


    def createMatrixes(self):
        self.distanceMatrix = np.zeros(shape = (self.nNodos, self.nNodos))
        self.timeMatrix = np.zeros(shape = (self.nNodos, self.nNodos))

        for i in range(0, self.nNodos):
            for j in range(0, self.nNodos):
                distance = np.linalg.norm(abs(self.n_coordenadas[j] - self.n_coordenadas[i]))
                self.distanceMatrix[i][j] = distance
                self.timeMatrix[i][j] = distance * 60 / 75


    def crearTW(self, twMin, twMax):
        if twMin is None:
            self.minTW = np.zeros(shape=self.nNodos)
        else:
            self.minTW = np.zeros(shape=self.nNodos) + twMin

        if twMax is None:
            self.maxTW = np.zeros(shape=self.nNodos) + float('inf')
        else:
            self.maxTW = np.zeros(shape=self.nNodos) + twMax



    # Guarda el último conjunto de grafos completado 
    def render(self):
        if self.singlePlot:
            self.grafoCompletado.guardarGrafosSinglePlot()

        else:
            self.grafoCompletado.guardarGrafos()

        self.crearReport(self.ordenVisitasCompletas, self.tiempoFinal)


    # Guarda el conjunto actual de grafos, independientemente de si están completos o no
    def render2(self):
        if self.singlePlot:
            self.rutas.guardarGrafosSinglePlot()
            
        else:
            self.rutas.guardarGrafos()
        
        self.crearReport(self.v_ordenVisitas, self.currTime)


    def crearReport(self, v_ordenVisitas, currTime, directorio = 'reports', name = 'report', extension = '.txt'):
        directorio = os.path.join(directorio, str(date.today()))
        
        if not os.path.exists(directorio):
            os.makedirs(directorio)

        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreDoc = os.path.join(directorio, name + '_' + siguiente_numero + extension)

        tiempoTotal = 0

        with open(nombreDoc, 'w', encoding="utf-8") as f:
            f.write("############")
            f.write(str(date.today()))
            f.write("############")
            f.write("\n\nNúmero de vehíclos utilizados: {}".format(self.nVehiculos))
            f.write("\n")

            for ruta, tiempo in zip(v_ordenVisitas, currTime):
                tiempoTotal += tiempo
                f.write("\n"+str(ruta) + " - t: " + str(round(tiempo, 2)) + " min")

            f.write("\n\nTiempo total: " + str(round(tiempoTotal, 2)) + " min")

            f.close()


    def actionParser(self, action):
        vehiculo = action // self.nNodos        
        action = action % self.nNodos

        return str(vehiculo) + "_" + str(action)
    

    def minToStr(self, time):
        hora = time / 60
        minutos = time // 60

        return str(hora) + ":" + str(minutos)