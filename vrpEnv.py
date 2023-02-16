from typing import Tuple, Union

from math import floor

import gym
from gym import spaces
import numpy as np

verbose = False

class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nVehiculos = 1, nNodos = 20, nGrafos = 10, nVisualizaciones = 5):
        self.step_count = 0
        self.nNodos = nNodos + 1 # +1 del depot
        self.nGrafos = nGrafos
        self.nVehiculos = nVehiculos
        self.nVisualizaciones = nVisualizaciones

        # Tantas acciones como (número de nodos + depot) * número de vehículos
        self.nActions = self.nNodos * self.nVehiculos

        print("nActions {}".format(self.nActions)) if verbose else None
        self.action_space = spaces.Discrete(self.nActions)

        self.observation_space = spaces.Dict({
            "visited" :  spaces.MultiDiscrete(np.zeros(shape=(self.nVehiculos, self.nNodos)) + 2),
            "curr_position" : spaces.MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.nNodos)
        })

    def step(self, action):
        self.step_count += 1
        
        # Comprobar si la acción es válida
        if not self.checkAction(action):
            return (
                self.getState(),
                -1,  # reward
                False,
                dict()
            )
        
        # supongamos que nNodos = 6, nVehiculos = 2 y action = 6 * 2 + 2
        # Calculamos el vehículo que realiza la acción

        vehiculo = action // self.nNodos
        # vehiculo = 1, es decir, el segundo vehículo
        print("action antes {}".format(action)) if verbose else None
        
        action = action % self.nNodos
        # action = 14 % 6 = 2 # Sería visitar el tercer nodo
        
        if verbose:
            print("nNodos {}".format(self.nNodos))
            print("action despues {}".format(action))
            print("vehiculo {}".format(vehiculo))

        # Eliminar el lugar que se acaba de visitar de las posibles acciones
        self.visited[:,action] = 1
        print(self.visited)  if verbose else None

        # Variar posición del vehículo que realice la acción
        self.posicionActual[vehiculo] = action
        print("posicionActual {}".format(self.posicionActual)) if verbose else None

        # Calcular la recompensa #TODO
        reward = 1

        # Comprobar si se ha llegado al final del entorno
        done = self.is_done()

        return (
            self.getState(),
            reward,
            done,
            dict()
        )


    def reset(self):
        self.step_count = 0
        self.visited = np.zeros(shape=(self.nVehiculos, self.nNodos))
        self.posicionActual = np.zeros(shape = self.nVehiculos)

        return self.getState()

    def checkAction(self, action) -> bool:
        return True
    
    def getState(self):
        obs = dict()
        obs["visited"] = self.visited
        obs["curr_position"] = self.posicionActual

        return obs

    def generate_mask(self):
        pass

    def is_done(self):
        return bool(np.all(self.visited == 1))
    
    def render(self):
        pass