from typing import Tuple, Union

import gym
import numpy as np

from rutas import Rutas

class VrpEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nVehiculos, nNodos, nGrafos, nVisualizaciones):
        assert (
                nVisualizaciones <= nGrafos
            ), "Num_draw needs to be equal or lower than the number of generated graphs."

        self.step_count = 0
        self.nNodos = nNodos
        self.nGrafos = nGrafos
        self.nVehiculos = nVehiculos
        self.nVisualizaciones = nVisualizaciones

        self.load = np.ones(shape=(nNodos,))


    def step(self, actions):
        self.step_count += 1

        # visit each next node
        self.visited[np.arange(len(actions)), actions.T] = 1
        traversed_edges = np.hstack([self.current_location, actions]).astype(int)
        self.sampler.visit_edges(traversed_edges)

        # get demand of the visited nodes
        selected_demands = self.demands[
            np.arange(len(self.demands)), actions.T
        ].squeeze()

        # update load of each vehicle
        self.load -= selected_demands
        self.load[np.where(actions == self.depots)[0]] = 1

        self.current_location = np.array(actions)

        if self.video_save_path is not None:
            self.vid.capture_frame()

        done = self.is_done()
        return (
            self.get_state(),
            -self.sampler.get_distances(traversed_edges),
            done,
            None,
        )

    def reset(self):
        self.step_count = 0
        
        # Ponemos los nodos visitados a 0
        self.visited = np.zeros(shape=(self.nGrafos, self.nNodos))
        self.sampler = Rutas(
            num_graphs=self.nGrafos, num_nodes=self.nNodos, num_depots=1, drawDemand = True,
        )

        # set current location to the depots
        self.depots = self.sampler.getDepots() # TODO
        self.current_location = self.depots

        self.demands = self.sampler.getDemands() # TODO

        self.load = np.ones(shape=(self.nGrafos,))
        
        return self.get_state() # TODO


    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:

        # generate state (depots not yet set)
        state = np.dstack(
            [
                self.sampler.getGraphPositions(),
                self.demands,
                np.zeros((self.nGrafos, self.nNodos)),
                self.generate_mask(),
            ]
        )

        # set depots in state to 1
        state[np.arange(len(state)), self.depots.T, 3] = 1

        return (state, self.load)

    def generate_mask(self):
        """
        Generates a mask of where the nodes marked as 1 cannot
        be visited in the next step according to the env dynamic.

        Returns:
            np.ndarray: Returns mask for each (un)visitable node
                in each graph. Shape (batch_size, num_nodes)
        """

        # disallow staying at the depot
        depot_graphs_idxs = np.where(self.current_location == self.depots)[0]
        self.visited[depot_graphs_idxs, self.depots[depot_graphs_idxs].squeeze()] = 1

        # allow visiting the depot when not currently at the depot
        depot_graphs_idxs_not = np.where(self.current_location != self.depots)[0]
        self.visited[
            depot_graphs_idxs_not, self.depots[depot_graphs_idxs_not].squeeze()
        ] = 0

        # allow staying on a depot if the graph is solved.
        done_graphs = np.where(np.all(self.visited, axis=1) == True)[0]
        self.visited[done_graphs, self.depots[done_graphs].squeeze()] = 0

        # disallow visiting nodes that exceed the current load.
        mask = np.copy(self.visited)
        exceed_demand_idxs = ((self.demands - self.load[:, None, None]) > 0).squeeze()
        mask[exceed_demand_idxs] = 1

        return mask

    def render(self, mode='human', close=False):
        pass