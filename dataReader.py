import pandas as pd
import numpy as np
import json
import sys
import os
import io


class DataReader:
    def __init__(self,
                 dataPath,
                 nodeFile="nodes",
                 vehicleFile="vehicles",
                 fileFormat = ".csv"
                 ):
        
        # Comprobamos si el directorio especificado en 'dataPath' existe
        if not os.path.exists(dataPath):
            print("Unable to find path: {}".format(dataPath), file=sys.stderr)
            sys.exit(1)  # Si no existe, salimos del programa

        self.nodePath = os.path.join(dataPath, nodeFile + fileFormat)
        self.vehiclePath = os.path.join(dataPath, vehicleFile + fileFormat)

        # Leer la información de nodos y vehículos
        if fileFormat == ".csv":
            self.nodeInfo = pd.read_csv(self.nodePath, index_col=0)
            self.vehicleInfo = pd.read_csv(self.vehiclePath, index_col=0)

        # Se usa para los benchmarks        
        elif fileFormat == ".json":
            with io.open(self.nodePath, 'rt', newline='') as file:
                jsonData = json.load(file)
            
            file.close()

            self.loadInfoJson(jsonData)

        else:
            print("The specified file format {} is not supported.".format(fileFormat), file=sys.stderr)



    def loadNodeData(self):
        nodeData = dict()

        nodeData['n_coordenadas'] = np.array([self.nodeInfo["coordenadas_X"], self.nodeInfo["coordenadas_Y"]]).T
        nodeData['n_originalDemands'] = self.nodeInfo["demandas"].to_numpy()
        nodeData['n_maxNodeCapacity ']= self.nodeInfo["maxDemand"][0]

        nodeData['minTW'] = self.nodeInfo["minTW"].to_numpy()
        nodeData['maxTW'] = self.nodeInfo["maxTW"].to_numpy()

        return nodeData


    def loadVehicleData(self):
        vehicleData = dict()

        vehicleData['v_maxCapacity'] = self.vehicleInfo["maxCapacity"][0]
        vehicleData['v_speed'] = self.vehicleInfo["speed"][0]
        vehicleData['v_loads'] = self.vehicleInfo["maxCapacity"].to_numpy()
        vehicleData['v_speeds'] = self.vehicleInfo["speed"].to_numpy()

        return vehicleData

    #def loadInfoJson(self, data):



if __name__ == "__main__":
    reader = DataReader('data', 'C101', 'lala', '.json')

    print(reader.nodeInfo)
