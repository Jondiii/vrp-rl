import time
import gym
from gym import spaces
import numpy as np
from rutas import Rutas
import copy
import os
from datetime import date
from dataGenerator import DataGenerator
from dataReader import DataReader


class VRPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    decayingStart = None
    grafoCompletado = None

    prev_action = 0
    prev_vehicle = 0

    def __init__(self, seed = 6, multiTrip = False, singlePlot = False):
        #np.random.seed(seed)
        self.multiTrip = multiTrip
        self.singlePlot = singlePlot
        self.currSteps = 0

        self.isDoneFunction = self.isDone

    def createEnv(self,
                  nVehiculos, nNodos, 
                  maxNumVehiculos = 50, maxNumNodos = 100,
                  maxCapacity = 100, maxNodeCapacity = 6,
                  sameMaxNodeVehicles = False,
                  twMin = None, twMax = None,
                  speed = 70
                  ):

        if sameMaxNodeVehicles:
            self.maxNumVehiculos = nVehiculos
            self.maxNumNodos = nNodos + 1

        else:
            self.maxNumVehiculos = maxNumVehiculos
            self.maxNumNodos = maxNumNodos + 1

        # Características del entorno
        self.nNodos = nNodos + 1 # +1 del depot
        self.nVehiculos = nVehiculos
        self.currTime = np.zeros(shape=(self.maxNumVehiculos,1))

        # Características de los nodos
        self.n_maxNodeCapacity = maxNodeCapacity
        self.twMin = twMin
        self.twMax = twMax

        # Características de los vehículos
        self.v_maxCapacity = maxCapacity
        self.v_speed = speed

        self.dataGen = DataGenerator(self.maxNumNodos, self.maxNumVehiculos)
        self.generateRandomData()
        
        # Cálculo de matrices de distancia
        self.createMatrixes()

        # Creamos el espacio de acciones y el espacio de observaciones
        self.createSpaces()



    def readEnvFromFile(self, nVehiculos, nNodos, filePath):
        self.dataReader  = DataReader(filePath)

        self.loadData(self.dataReader)
        
        self.nNodos = nNodos + 1
        self.nVehiculos = nVehiculos

        self.maxNumNodos = len(self.dataReader.nodeInfo.index)
        self.maxNumVehiculos = len(self.dataReader.vehicleInfo.index)

        # Cálculo de matrices de distancia
        self.createMatrixes()

        # Creamos el espacio de acciones y el espacio de observaciones
        self.createSpaces()



    def createSpaces(self):
        # Tantas acciones como (número de nodos + depot) * número de vehículos
        self.action_space = spaces.Discrete(self.maxNumNodos * self.maxNumVehiculos - 1)

        # Aquí se define cómo serán las observaciones que se pasarán al agente.
        # Se usa multidiscrete para datos que vengan en formato array. Hay que definir el tamaño de estos arrays
        # y sus valores máximos. La primera, "visited", podrá tomar un máximo de 2 valores en cada posición del array
        # (1 visitado - 0 sin visitar), por lo que le pasamos un array [2,2,...,2]
        self.observation_space = spaces.Dict({
            "n_visited" :  spaces.MultiDiscrete(np.zeros(shape=self.maxNumNodos) + 2), # TODO: poner como multi binary??
            "v_curr_position" : spaces.MultiDiscrete(np.zeros(shape=self.maxNumVehiculos) + self.maxNumNodos),
            "v_loads" : spaces.MultiDiscrete(np.zeros(shape=self.maxNumVehiculos) + self.v_maxCapacity + 1), # SOLO se pueden usar enteros
            "n_demands" : spaces.MultiDiscrete(np.zeros(shape=self.maxNumNodos) + self.n_maxNodeCapacity * 5),
            #"v_curr_time" : spaces.Box(low = 0, high = float('inf'), shape = (self.maxNumVehiculos,), dtype=float),
            "n_distances" : spaces.Box(low = 0, high = float('inf'), shape = (self.maxNumVehiculos * self.maxNumNodos,), dtype=float),
            #"n_timeLeftTWClose" : spaces.Box(low = float('-inf'), high = float('inf'), shape = (self.maxNumVehiculos * self.maxNumNodos,), dtype=float) # Con DQN hay que comentar esta línea
        })


    def step(self, action):
        self.currSteps += 1
        if action >= self.nNodos * self.nVehiculos:
            return self.getState(), -1, self.isDoneFunction(), dict(info = "Acción rechazada por actuar sobre un nodo no disponible.", accion = action, nNodos = self.nNodos)

        # supongamos que nNodos = 6, nVehiculos = 2 y action = 6 * 2 + 2
        # Calculamos el vehículo que realiza la acción
        vehiculo = action // self.nNodos
        # vehiculo = 1, es decir, el segundo vehículo
        action = action % self.nNodos
        # action = 14   %  6 = 2 # Sería visitar el tercer nodo

        # Comprobar si la acción es válida
        if not self.checkAction(action, vehiculo):
            return self.getState(), -1, self.isDoneFunction(), dict()

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
        reward = self.getReward(distancia, action, vehiculo)

        # Actualizar posición del vehículo que realice la acción.
        self.v_posicionActual[vehiculo] = action

        # Añadir el nodo a la ruta del vehículo.
        self.v_ordenVisitas[vehiculo].append(action)

        # Actualizar las distancias a otros nodos
        self.n_distances[vehiculo] = self.distanceMatrix[action]

        # Actualizar tiempo de recorrido del vehículo que realice la acción
        self.currTime[vehiculo] += tiempo

        #self.graphicalRender()
        self.prev_action = action
        self.prev_vehicle = vehiculo

        # Comprobar si se ha llegado al final del episodio
        done = self.isDoneFunction()

        return self.getState(), reward, done, dict()



    def reset(self):
        self.visited = np.zeros(shape=(self.nNodos))
        self.visited = np.pad(self.visited, (0,self.maxNumNodos - self.nNodos), 'constant', constant_values = 1)

        if self.multiTrip:
            self.visited[0] = 0
        else:
            self.visited[0] = 1 # El depot comienza como visitado

        self.v_posicionActual = np.zeros(shape = self.maxNumVehiculos)

        self.v_loads = np.zeros(shape=self.nVehiculos,) + self.v_maxCapacity
        self.v_loads = np.pad(self.v_loads, (0, self.maxNumVehiculos - self.nVehiculos), 'constant', constant_values = 0)

        self.n_demands = copy.deepcopy(self.n_originalDemands)
        self.currTime = np.zeros(shape=(self.nVehiculos), dtype = float)
        self.currTime = np.pad(self.currTime, (0, self.maxNumVehiculos - self.nVehiculos), 'constant', constant_values = 999)

        self.n_distances = np.zeros(shape = (self.maxNumVehiculos, self.maxNumNodos))

        for i in range(0, self.maxNumVehiculos):
            self.n_distances[i] = self.distanceMatrix[0]

        # Creamos un conjunto de rutas nuevo

        self.rutas = Rutas(self.nVehiculos, self.nNodos, self.maxNumVehiculos, self.maxNumNodos, self.n_demands, self.n_coordenadas, self.v_speeds)
        
        self.v_ordenVisitas = []

        for _ in range(self.nVehiculos):
            self.v_ordenVisitas.append([0])
        
        self.done = False
        
        return self.getState()



    def checkAction(self, action, vehiculo):
        if self.visited[action] == 1:
            return False
        
        if self.v_posicionActual[vehiculo] == action:
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
        obs["n_demands"] = self.n_demands 
        #obs["v_curr_time"] = self.currTime
        obs["n_distances"] = self.n_distances.flatten()
        #obs["n_timeLeftTWClose"] = self.getTimeLeftTWClose().flatten()

        return obs


    def setIncreasingIsDone(self, totalSteps, minNumVisited = 0.5, increaseStart = 0.5, increaseRate = 0.1, everyNtimesteps = 0.1):
        self.totalSteps = totalSteps
        self.increaseStart = increaseStart
        self.increaseRate = increaseRate
        self.everyNTimesteps = totalSteps * everyNtimesteps

        self.minimumVisited = minNumVisited

        self.isDoneFunction = self.increasingIsDone



    def increasingIsDone(self):
        if self.currSteps / self.totalSteps >= self.increaseStart:
            if self.currSteps % self.everyNTimesteps == 0: 
                self.minimumVisited += self.increaseRate

                if self.minimumVisited >= 1:
                    self.minimumVisited = 1


        if self.multiTrip:
            porcentajeVisitados = np.count_nonzero(self.visited[1:self.nNodos] == 1) / self.nNodos

            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.currTime)
                return True
    

        if np.all(self.v_posicionActual == 0):
            porcentajeVisitados = np.count_nonzero(self.visited[:self.nNodos] == 1) / self.nNodos
            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.currTime)
                return True
            
        else:
            porcentajeVisitados = np.count_nonzero(self.visited[:self.nNodos] == 1) / self.nNodos
            
            if porcentajeVisitados >= self.minimumVisited:
                self.visited[0] = 0

        return False



    def setDecayingIsDone(self, totalSteps, decayingStart = 0.5, decayingRate = 0.1, everyNtimesteps = 0.05):
        self.totalSteps = totalSteps
        self.decayingStart = decayingStart
        self.decayingRate = decayingRate
        self.everyNTimesteps = totalSteps * everyNtimesteps

        self.minimumVisited = 1

        self.isDoneFunction = self.decayingIsDone


    def decayingIsDone(self):
        if self.minimumVisited == 1:
            if self.currSteps / self.totalSteps >= self.decayingStart: # Si se pasa de más de (50%), hacemos que solo haya que visitar el (90%)
                self.minimumVisited -= self.decayingRate

            return self.isDone()
        
        if self.currSteps % self.everyNTimesteps == 0: # Si hace
            self.minimumVisited -= self.decayingRate

        if self.multiTrip:
            porcentajeVisitados = np.count_nonzero(self.visited[1:self.nNodos] == 1) / self.nNodos

            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.currTime)
                return True
        

        if np.all(self.v_posicionActual == 0):
            porcentajeVisitados = np.count_nonzero(self.visited[:self.nNodos] == 1) / self.nNodos
                        
            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.currTime)
                return True
            
        else:
            porcentajeVisitados = np.count_nonzero(self.visited[:self.nNodos] == 1) / self.nNodos
            
            if porcentajeVisitados >= self.minimumVisited:
                self.visited[0] = 0

     
        return False

        




    def isDone(self): # can't DO: cambiar esto de orden, primero comprobar vehículo y después nodos --> no se puede por lo de marcar el depot como no visitado
        if self.multiTrip:
            allVisited = np.all(self.visited[1:] == 1) # TODO Aquí da igual que el camión haya vuelto al depot o no... está bien??

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


    def getReward(self, distancia, action, vehicle):
        if action == self.prev_action:
            if vehicle == self.prev_vehicle:
                return -5
            
        if distancia == 0:
            return -5

        reward = round(1/abs(distancia), 2)

        if self.visited[0] == 0:
            reward += 1 # La idea detrás de esto es recompensar que los vehículos vuelvan al depot
            pass
        
        return reward



    # Current time tiene shape (nVehiculos,), mientras que el repeat devuelve shape (nNodos, nvehiculos)
    # No se puede hacer broadcast de esos rangos, pero con np.array([self.currTime]).T tenemos que curr time tiene shape (nVehiculos,1)
    # y con eso sí se puede hacer la resta que queremos
    def getTimeLeftTWClose(self):
        twClose = np.repeat([self.maxTW], repeats=self.maxNumVehiculos, axis=0) - np.array([self.currTime]).T

        return twClose



    def createMatrixes(self):
        self.distanceMatrix = np.zeros(shape = (self.maxNumNodos, self.maxNumNodos))
        self.timeMatrix = np.zeros(shape = (self.maxNumNodos, self.maxNumNodos))
        for i in range(0, self.maxNumNodos):
            for j in range(0, self.maxNumNodos):
                distance = np.linalg.norm(abs(self.n_coordenadas[j] - self.n_coordenadas[i]))
                self.distanceMatrix[i][j] = distance
                self.timeMatrix[i][j] = distance * 60 / 75



    def crearTW(self, twMin, twMax):
        if twMin is None:
            self.minTW = np.zeros(shape=self.maxNumNodos)
        else:
            self.minTW = np.zeros(shape=self.maxNumNodos) + twMin

        if twMax is None:
            self.maxTW = np.zeros(shape=self.maxNumNodos) + 100000 # No deja poner inf, tensorflow lo interpreta como nan
        else:
            self.maxTW = np.zeros(shape=self.maxNumNodos) + twMax



    def generateRandomData(self):
        self.dataGen.addNodeInfo(self.n_maxNodeCapacity, self.twMin, self.twMin)
        self.dataGen.generateNodeInfo()
        self.dataGen.addVehicleInfo(self.v_maxCapacity, self.v_speed)
        self.dataGen.generateVehicleInfo()
        self.dataGen.saveData()

        self.loadData(self.dataGen)


    def loadData(self, reader):
        # Características de los nodos
        self.n_coordenadas = np.array([reader.nodeInfo["coordenadas_X"], reader.nodeInfo["coordenadas_Y"]]).T

        self.n_originalDemands = reader.nodeInfo["demandas"].to_numpy()
        self.n_demands = copy.deepcopy(self.n_originalDemands)
        self.n_maxNodeCapacity = reader.nodeInfo["maxDemand"][0] # TODO

        # Características de los vehículos
        self.v_maxCapacity = reader.vehicleInfo["maxCapacity"][0] # TODO
        self.v_speed = reader.vehicleInfo["speed"][0]      # TODO
        self.v_loads = reader.vehicleInfo["maxCapacity"].to_numpy()
        self.v_speeds = reader.vehicleInfo["speed"].to_numpy()

        self.minTW = reader.nodeInfo["minTW"].to_numpy()
        self.maxTW = reader.nodeInfo["maxTW"].to_numpy()



    # Guarda el último conjunto de grafos completado 
    def render(self, fecha = None):
        if self.grafoCompletado == None:
            return
        
        if self.singlePlot:
            self.grafoCompletado.guardarGrafosSinglePlot(fecha)

        else:
            self.grafoCompletado.guardarGrafos(fecha)

        self.crearReport(self.ordenVisitasCompletas, self.tiempoFinal, fecha)


    # Guarda el conjunto actual de grafos, independientemente de si están completos o no
    def render2(self):
        if self.singlePlot:
            self.rutas.guardarGrafosSinglePlot()
        else:
            self.rutas.guardarGrafos()
        
        self.crearReport(self.v_ordenVisitas, self.currTime)



    def graphicalRender(self):
        self.rutas.getRutasVisual()


    def crearReport(self, v_ordenVisitas, currTime, fecha, directorio = 'reports', name = 'report', extension = '.txt'):
        if fecha is None:
            fecha = str(date.today())

        directorio = os.path.join(directorio, fecha)

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
