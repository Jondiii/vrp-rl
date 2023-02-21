import gym
from gym import spaces
import numpy as np
from grafo import Grafo

class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nVehiculos = 1, nNodos = 20, nGrafos = 10, nVisualizaciones = 5, maxCapacity = 100):
        self.step_count = 0
        
        self.nNodos = nNodos + 1 # +1 del depot
        self.nVehiculos = nVehiculos

        self.nGrafos = nGrafos
        self.nVisualizaciones = nVisualizaciones
        
        self.maxCapacity = maxCapacity
        self.demands = np.random.randint(low = 1, high = 30, size=self.nNodos)
        self.coordenadas = np.random.rand(nNodos+1, 2) # [0, nNodos), por lo que hay que sumarle +1

        self.loads = np.zeros(shape=self.nVehiculos) + self.maxCapacity

        # Tantas acciones como (número de nodos + depot) * número de vehículos
        self.nActions = self.nNodos * self.nVehiculos

        self.action_space = spaces.Discrete(self.nActions)

        self.observation_space = spaces.Dict({
            "visited" :  spaces.MultiDiscrete(np.zeros(shape=(self.nNodos)) + 2),
            "curr_position" : spaces.MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.nNodos)
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
        self.grafo.visitEdge(self.posicionActual[vehiculo], action)

        # Calcular la recompensa #TODO
        reward = 1
        print(self.posicionActual)
        # Variar posición del vehículo que realice la acción
        self.posicionActual[vehiculo] = action

        # Comprobar si se ha llegado al final del entorno
        done = self.is_done()
        print(done)
        return self.getState(), reward, done, dict()

    def reset(self):
        self.step_count = 0
        self.visited = np.zeros(shape=(self.nNodos))
        self.posicionActual = np.zeros(shape = self.nVehiculos)
        self.loads = np.zeros(shape=self.nVehiculos) + self.maxCapacity

        # Creamos un grafo nuevo
        self.grafo = Grafo(self.nNodos, self.demands, self.coordenadas)

        self.done = False

        return self.getState()

    def checkAction(self, action, vehiculo) -> bool:
        if self.visited[action] == 1:
            return False
        
        if self.loads[vehiculo] - self.demands[action] < 0:
            return False

        return True
    
    def getState(self):
        obs = dict()
        obs["visited"] = self.visited
        obs["curr_position"] = self.posicionActual

        return obs

    # TODO
    def generate_mask(self):
        pass


    def is_done(self):
        return bool(np.all(self.visited == 1))
    
    def render(self):
        self.grafo.guardarGrafo()