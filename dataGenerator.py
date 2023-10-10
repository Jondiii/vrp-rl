import pandas as pd
import numpy as np
import os

"""
Esta clase es la encargada de generar datos semi-aleatorios con los que se han ido haciendo pruebas durante el desarrollo
"""

class DataGenerator:

    """
    numNodos = Número de nodos del problema
    numVehiculos = Número de vehículos del problema
    dataPath = Directorio donde se encuentran los datos
    nodeFile = Nombre del fichero donde se encuentran los datos de los nodos
    vehicleFile = Nombre del fichero donde se encuentran los datos de los vehículos
    seed = Semilla para la generación semi-aleatoria
    """
    def __init__(self,
                 numNodos,
                 numVehiculos,
                 dataPath = "data",
                 nodeFile = "nodes.csv",
                 vehicleFile = "vehicles.csv",
                 seed = None
                 ):
        
        if seed is not None:
            np.random.seed(seed)

        if not os.path.exists(dataPath):
            os.makedirs(dataPath)

        self.numNodos = numNodos
        self.numVehiculos = numVehiculos

        self.nodePath = os.path.join(dataPath, nodeFile)
        self.vehiclePath = os.path.join(dataPath, vehicleFile)

    """
    Método que añade información acerca de los nodos. La información a añadir es:
        n_MaxDemands = Valor máximo de la demanda del nodo. Este valor es después multiplicado por 5, de manera que
                       los valores aleatorios resultantes sean siempre números enteros.
        n_twMin = Valor mínimo de las ventanas de tiempo de los nodos. 
        n_twMax = Valor máximo de las ventanas de tiempo de los nodos. 
    """ 
    def addNodeInfo(self, n_MaxDemands = 6, n_twMin = None, n_twMax =  None):
        self.n_MaxDemands = n_MaxDemands
        self.n_twMin = n_twMin
        self.n_twMax = n_twMax


    """
    Método que añade información acerca de los vehñiculos. La información a añadir es:
        v_MaxDemands = Valor máximo de la carga del vehículo. Este valor NO se multiplica por 5.
        v_speed = Velocidad de los vehículos que se asume que es constante durante toda la ruta. 
    """ 
    def addVehicleInfo(self, v_MaxDemands = 100, v_speed = 70):
        self.v_MaxCapacity = v_MaxDemands
        self.v_speed = v_speed



    # Método para generar información aleatoria de nodos
    def generateNodeInfo(self):
        # Crear un DataFrame vacío para almacenar la información de los nodos
        self.nodeInfo = pd.DataFrame()

        # Establecer el nombre de índice como "index"
        self.nodeInfo.index.name = "index"

        # Generar coordenadas X e Y aleatorias comprendidas en el rango [0,1) para cada nodo
        self.nodeInfo["coordenadas_X"] = np.random.rand(self.numNodos)
        self.nodeInfo["coordenadas_Y"] = np.random.rand(self.numNodos)

        # Generar demandas aleatorias para cada nodo (multiplicando por 5)
        self.nodeInfo["demandas"] = (np.random.randint(low=1, high=self.n_MaxDemands, size=self.numNodos) * 5)

        # Establecer la capacidad máxima de demanda para todos los nodos
        self.nodeInfo["maxDemand"] = self.n_MaxDemands * 5

        # Establecer las time windows mínimas y máximas para cada nodo
        # Si no se especifica, se usa 0 como mínimo y 100.000 como máximo, de manera que siempre se cumplan
        self.nodeInfo["minTW"] = 0 if self.n_twMin is None else self.n_twMin
        self.nodeInfo["maxTW"] = 100000 if self.n_twMax is None else self.n_twMax


    # Método para generar información de vehículos
    def generateVehicleInfo(self):
        # Crear un DataFrame vacío para almacenar la información de los vehículos
        self.vehicleInfo = pd.DataFrame()

        # Establecer el nombre de índice como "index"
        self.vehicleInfo.index.name = "index"

        # Inicializar la posición inicial de todos los vehículos en 0
        self.vehicleInfo["initialPosition"] = np.zeros(self.numVehiculos, dtype=int)

        # Establecer la capacidad máxima de carga para todos los vehículos
        self.vehicleInfo["maxCapacity"] = self.v_MaxCapacity

        # Establecer la velocidad media para todos los vehículos
        self.vehicleInfo["speed"] = self.v_speed


    # Método para guardar los datos de nodos y vehículos en archivos CSV
    def saveData(self):
        self.nodeInfo.to_csv(self.nodePath)
        self.vehicleInfo.to_csv(self.vehiclePath)

    # Método para leer los datos de nodos y vehículos desde archivos CSV (creo que no se usa)
    def readData(self):
        self.nodeInfo = pd.read_csv(self.nodePath, index_col=0)

        self.vehicleInfo = pd.read_csv(self.vehiclePath, index_col=0)


# Esto era para hacer pruebas durante el desarrollo. No molesta, así que no se ha boorado.
if __name__ == "__main__":
    dataGen = DataGenerator(20, 5)
    dataGen.addNodeInfo()
    dataGen.generateNodeInfo()
    dataGen.addVehicleInfo()
    dataGen.generateVehicleInfo()
    dataGen.saveData()
    prueba = np.array([dataGen.nodeInfo["coordenadas_X"], dataGen.nodeInfo["coordenadas_Y"]])
    prueba2 = dataGen.nodeInfo["demandas"].to_numpy()
    print(prueba2)

