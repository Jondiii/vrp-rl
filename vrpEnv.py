import gym
from gym import spaces
import numpy as np
from rutas import Rutas
import copy

class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nVehiculos = 1, nNodos = 20, nGrafos = 10, nVisualizaciones = 5, maxCapacity = 100, maxNodeCapacity = 30):
        self.step_count = 0
        
        self.nNodos = nNodos + 1 # +1 del depot
        self.nVehiculos = nVehiculos

        self.nGrafos = nGrafos
        self.nVisualizaciones = nVisualizaciones
        
        self.maxCapacity = maxCapacity
        self.maxNodeCapacity = maxNodeCapacity
        self.demands = np.random.randint(low = 1, high = maxNodeCapacity, size=self.nNodos)
        self.coordenadas = np.random.rand(nNodos+1, 2) # [0, nNodos), por lo que hay que sumarle +1

        self.loads = np.zeros(shape=self.nVehiculos) + self.maxCapacity

        # Tantas acciones como (número de nodos + depot) * número de vehículos
        self.nActions = self.nNodos * self.nVehiculos

        self.action_space = spaces.Discrete(self.nActions)

        # Aquí se define cómo serán las observaciones que se pasarán al agente.
        # Se usa multidiscrete para datos que vengan en formato array. Hay que definir el tamaño de estos arrays
        # y sus valores máximos. La primera, "visited", podrá tomar un máximo de 2 valores en cada posición del array
        # (1 visitado - 0 sin visitar), por lo que le pasamos un array [2,2,...,2]
        self.observation_space = spaces.Dict({
            "visited" :  spaces.MultiDiscrete(np.zeros(shape=self.nNodos) + 2),
            "curr_position" : spaces.MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.nNodos),
            "vehicle_loads" : spaces.MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.maxCapacity + 1), # SOLO se pueden usar enteros
            "demands" : spaces.MultiDiscrete(np.zeros(shape=self.nNodos) + self.maxNodeCapacity)
        })

    def step(self, action):
        self.step_count += 1
    
        # supongamos que nNodos = 6, nVehiculos = 2 y action = 6 * 2 + 2
        # Calculamos el vehículo que realiza la acción
        vehiculo = action // self.nNodos
        # vehiculo = 1, es decir, el segundo vehículo
        
        action = action % self.nNodos
        # action = 14   %  6 = 2 # Sería visitar el tercer nodo

        # Comprobar si la acción es válida
        if not self.checkAction(action, vehiculo):
            return self.getState(), -1, False, dict()
        
        self.loads[vehiculo] -= self.demands[action] # Añadimos la demanda del nodo a la carga del vehículo

        # Eliminar el lugar que se acaba de visitar de las posibles acciones
        self.visited[action] = 1

        # Marcamos la visita en el grafo
        self.rutas.visitEdge(vehiculo, self.posicionActual[vehiculo], action)
        
        # Calcular la recompensa
        reward = abs(self.rutas.getDistance(vehiculo, self.posicionActual[vehiculo], action))

        # Variar posición del vehículo que realice la acción
        self.posicionActual[vehiculo] = action

        # Comprobar si se ha llegado al final del entorno
        done = self.is_done()

        return self.getState(), reward, done, dict()

    def reset(self):
        self.step_count = 0
        self.visited = np.zeros(shape=(self.nNodos))
        self.visited[0] = 1 # El depot comienza como visitado
        self.posicionActual = np.zeros(shape = self.nVehiculos)
        self.loads = np.zeros(shape=self.nVehiculos,) + self.maxCapacity

        # Creamos un grafo nuevo
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
        obs["visited"] = self.visited
        obs["curr_position"] = self.posicionActual
        obs["vehicle_loads"] = self.loads
        obs["demands"] = self.demands   # No sé si tiene mucho sentido pasarle la demanda cuando esta no va a cambiar...
                                        # A no ser que pongamos la demanda de un nodo a 0 cuando esta sea recogida.
        return obs

    def is_done(self):
        allVisited = bool(np.all(self.visited == 1))

        if allVisited and bool(np.all(self.posicionActual == 0)):
            self.grafoCompletado = copy.deepcopy(self.rutas)
            return True
        
        elif allVisited:# Marcamos el depot como "no visitado", para que sea la única acción posible y tengan que volver al final del recorrido
            self.visited[0] = 0 

        return False
    
    def render(self):
        self.grafoCompletado.guardarGrafos()

    def render2(self):
        self.rutas.guardarGrafos()