import gym
from gym import spaces
import numpy as np
from rutas import Rutas
import copy

class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nVehiculos, nNodos, maxCapacity = 100, maxNodeCapacity = 30, seed = 6, multiTrip = False, singlePlot = False):        
        np.random.seed(seed)

        self.multiTrip = multiTrip
        self.singlePlot = singlePlot

        self.nNodos = nNodos + 1 # +1 del depot
        self.nVehiculos = nVehiculos
        self.coordenadas = np.random.rand(nNodos+1, 2) # [0, nNodos), por lo que hay que sumarle +1

        self.maxCapacity = maxCapacity
        self.maxNodeCapacity = maxNodeCapacity
        self.originalDemands = np.random.randint(low = 1, high = maxNodeCapacity, size=self.nNodos)
        self.demands = copy.deepcopy(self.originalDemands)
        
        self.loads = np.zeros(shape=self.nVehiculos) + self.maxCapacity

        # Tantas acciones como (número de nodos + depot) * número de vehículos
        self.nActions = self.nNodos * self.nVehiculos

        self.action_space = spaces.Discrete(self.nActions)

        # Aquí se define cómo serán las observaciones que se pasarán al agente.
        # Se usa multidiscrete para datos que vengan en formato array. Hay que definir el tamaño de estos arrays
        # y sus valores máximos. La primera, "visited", podrá tomar un máximo de 2 valores en cada posición del array
        # (1 visitado - 0 sin visitar), por lo que le pasamos un array [2,2,...,2]
        self.observation_space = spaces.Dict({
            "n_visited" :  spaces.MultiDiscrete(np.zeros(shape=self.nNodos) + 2),
            "v_curr_position" : spaces.MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.nNodos),
            "v_loads" : spaces.MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.maxCapacity + 1), # SOLO se pueden usar enteros
            "n_demands" : spaces.MultiDiscrete(np.zeros(shape=self.nNodos) + self.maxNodeCapacity)
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
                self.loads[vehiculo] = self.maxCapacity
                self.visited[action] = 0 # Si lo que se ha visitado es el depot, no lo marcamos como visitado 

        self.loads[vehiculo] -= self.demands[action] # Añadimos la demanda del nodo a la carga del vehículo


        # Marcamos la visita en el grafo
        self.rutas.visitEdge(vehiculo, self.posicionActual[vehiculo], action)

        # Ponemos la demanda del nodo a 0
        self.demands[action] = 0
        
        # Calcular la recompensa. Será inversamente proporcional a la distancia recorrida. 
        reward = self.getReward(vehiculo, action)

        # Variar posición del vehículo que realice la acción.
        self.posicionActual[vehiculo] = action

        # Comprobar si se ha llegado al final del episodio
        done = self.isDone()

        return self.getState(), reward, done, dict()

    def reset(self):
        self.visited = np.zeros(shape=(self.nNodos))
        
        if self.multiTrip:
            self.visited[0] = 0
        else:
            self.visited[0] = 1 # El depot comienza como visitado

        self.posicionActual = np.zeros(shape = self.nVehiculos)
        self.loads = np.zeros(shape=self.nVehiculos,) + self.maxCapacity
        self.demands = copy.deepcopy(self.originalDemands)

        # Creamos un conjunto de rutas nuevo
        self.rutas = Rutas(self.nVehiculos, self.nNodos, self.demands, self.coordenadas)

        self.done = False

        return self.getState()

    def checkAction(self, action, vehiculo):
        if self.visited[action] == 1:
            return False
        
        if self.loads[vehiculo] - self.demands[action] < 0:
            return False

        return True
    
    def getState(self):
        obs = dict()
        obs["n_visited"] = self.visited
        obs["v_curr_position"] = self.posicionActual
        obs["v_loads"] = self.loads
        obs["n_demands"] = self.demands   # No sé si tiene mucho sentido pasarle la demanda cuando esta no va a cambiar...
                                        # A no ser que pongamos la demanda de un nodo a 0 cuando esta sea recogida.
        return obs

    def isDone(self): # can't DO: cambiar esto de orden, primero comprobar vehículo y después nodos --> no se puede por lo de marcar el depot como no visitado
        if self.multiTrip:
            allVisited = np.all(self.visited[1:] == 1)

            if allVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                return True

        if np.all(self.posicionActual == 0):
            allVisited = np.all(self.visited == 1)
            if allVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                return True
        else:
            allVisited = np.all(self.visited == 1)
            if allVisited:
                self.visited[0] = 0 
        
        return False
    
    def getReward(self, vehiculo, action):
        distancia = abs(self.rutas.getDistance(vehiculo, self.posicionActual[vehiculo], action))
        reward = round(1/distancia, 2)

        if distancia == 0:
            reward = 0
        
        return reward

    # Guarda el último conjunto de grafos completado 
    def render(self):
        if self.singlePlot:
            self.grafoCompletado.guardarGrafosSinglePlot()

        else:
            self.grafoCompletado.guardarGrafos()

    # Guarda el conjunto actual de grafos, independientemente de si están completos o no
    def render2(self):
        if self.singlePlot:
            self.rutas.guardarGrafosSinglePlot()
            
        else:
            self.rutas.guardarGrafos()
            
    def actionParser(self, action):
        vehiculo = action // self.nNodos        
        action = action % self.nNodos

        return str(vehiculo) + "_" + str(action)
