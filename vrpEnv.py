from typing import Tuple, Union

from math import floor

import gym
from gym import spaces
import numpy as np

from rutas import Rutas

class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nVehiculos = 1, nNodos = 20, nGrafos = 10, nVisualizaciones = 5):
        self.step_count = 0
        self.nNodos = nNodos
        self.nGrafos = nGrafos
        self.nVehiculos = nVehiculos
        self.nVisualizaciones = nVisualizaciones

        # Tantas acciones como (número de nodos + depot) * número de vehículos
        nActions = (nNodos + 1) * nVehiculos

        self.action_space = spaces.Discrete(nActions)
        self.actions = np.arrange(stop = nActions)

    def step(self, action):
        self.step_count += 1
        
        # Comprobar si la acción es válida
        if not self.checkAction(action):
            return (
                self.getState(),
                -1,  # reward
                False,
                None
            )
        
        # supongamos que nNodos = 6, nVehiculos = 2 y action = 14

        vehiculo = floor(action % self.nNodos) # Calculamos el vehículo que realiza la acción
        # vehiculo = 1, es decir, el segundo vehículo

        action -= (self.nNodos + 1) * vehiculo
        # action = 14 - (6 + 1) * 1 = 7
        
        # Eliminar el lugar que se acaba de visitar de las posibles acciones
        self.visited[:,action] = 1

        # Variar posición del vehículo que realice la acción
        self.posicionActual[vehiculo] = action

        # Calcular la recompensa #TODO
        reward = 1

        # Comprobar si se ha llegado al final del entorno
        done = self.is_done()

        return (
            self.getState(),
            reward,
            done,
            None
        )

    def reset(self):
        self.step_count = 0

        self.visited = np.zeros(shape=(self.nVehiculos, self.nNodos))

        self.posicionActual = np.zeros(self.nVehiculos)

        return self.getState()

    def checkAction(self, actions) -> bool:
        pass
    
    def getState(self):
        pass

    def generate_mask(self):
        pass

    def is_done(self):
        return np.all(self.visited == 1)
    
    def render(self):
        pass