import pandas as pd
import numpy as np
import os

class DataGenerator:

    def __init__(self,
                 numNodos,
                 numVehiculos,
                 dataPath = "data",
                 nodeFile = "nodes.csv",
                 vehicleFile = "vehicles.csv",
                 seed = 6
                 ):
        
        np.random.seed(seed)
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)

        self.numNodos = numNodos
        self.numVehiculos = numVehiculos

        self.nodePath = os.path.join(dataPath, nodeFile)
        self.vehiclePath = os.path.join(dataPath, vehicleFile)



    def addNodeInfo(self, n_MaxDemands = 6, n_twMin = None, n_twMax =  None):
        self.n_MaxDemands = n_MaxDemands
        self.n_twMin = n_twMin
        self.n_twMax = n_twMax


    def addVehicleInfo(self, v_MaxDemands = 100, v_speed = 70):
        self.v_MaxCapacity = v_MaxDemands
        self.v_speed = v_speed



    def generateNodeInfo(self):
        self.nodeInfo = pd.DataFrame()

        self.nodeInfo.index.name = "index"

        self.nodeInfo["coordenadas_X"] = np.random.rand(self.numNodos)
        self.nodeInfo["coordenadas_Y"] = np.random.rand(self.numNodos)
        
        self.nodeInfo["demandas"] = (np.random.randint(low = 1, high = self.n_MaxDemands, size =  self.numNodos) * 5)
        self.nodeInfo["maxDemand"] = self.n_MaxDemands * 5

        self.nodeInfo["minTW"] = 0 if self.n_twMin is None else self.n_twMin
        self.nodeInfo["maxTW"] = 100000 if self.n_twMax is None else self.n_twMax


    def generateVehicleInfo(self):
        self.vehicleInfo = pd.DataFrame()

        self.vehicleInfo.index.name = "index"

        self.vehicleInfo["initialPosition"] = np.zeros(self.numVehiculos, dtype=int)
        self.vehicleInfo["maxCapacity"] = self.v_MaxCapacity
        self.vehicleInfo["speed"] = self.v_speed


    def saveData(self):
        self.nodeInfo.to_csv(self.nodePath)
        self.vehicleInfo.to_csv(self.vehiclePath)

    
    def readData(self):
        self.nodeInfo = pd.read_csv(self.nodePath, index_col=0)
        self.vehicleInfo = pd.read_csv(self.vehiclePath, index_col=0)

   
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

