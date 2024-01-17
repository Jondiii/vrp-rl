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


    def loadInfoJson(self, data):
        self.nodeInfo = self.loadNodeInfoJson(data, 'customer')
        self.vehicleInfo = self.loadVehicleInfoJson(data)



    def loadNodeInfoJson(self, data, keyword):
        nodeData = dict()
        self.nodeInfo = []
        
        value = data['depart']
        value['coordenadas_X'] = value['coordinates']['x']
        value['coordenadas_Y'] = value['coordinates']['y']

        self.nodeInfo.append(data['depart'])
        
        for key, value in data.items():
            if keyword in key:
                value['coordenadas_X'] = value['coordinates']['x']
                value['coordenadas_Y'] = value['coordinates']['y']
                self.nodeInfo.append(value)

        self.nodeInfo = pd.DataFrame.from_dict(self.nodeInfo)

        nodeData['index'] = self.nodeInfo.index

        nodeData['coordenadas_X'] = self.nodeInfo['coordenadas_X']
        nodeData['coordenadas_Y'] = self.nodeInfo['coordenadas_Y']
        
        nodeData['demandas'] = self.nodeInfo['demand']
        nodeData['maxDemand'] = self.nodeInfo['demand'].max()
        
        nodeData['minTW'] = self.nodeInfo["ready_time"].to_numpy()
        nodeData['maxTW'] = self.nodeInfo["due_time"].to_numpy()

        nodeData['service_time'] = self.nodeInfo["service_time"].to_numpy()

        return nodeData
    

    def loadVehicleInfoJson(self, data):
        vehicleData = pd.DataFrame(np.zeros(data['max_vehicle_number']))

        vehicleData.drop(columns=0, inplace=True)

        vehicleData['index'] = vehicleData.index

        vehicleData['initialPosition'] = 0
        vehicleData['maxCapacity'] = int(data["vehicle_capacity"])
        vehicleData['speed'] = 70

    
        return vehicleData.to_dict()



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
    

    def save_to_csv(self, path, nodeFile, vehicleFile):
        if not os.path.exists(path):
            os.makedirs(path)
        
        pd.DataFrame(self.nodeInfo).set_index('index').to_csv(os.path.join(path, nodeFile + '.csv'))
        pd.DataFrame(self.vehicleInfo).set_index('index').to_csv(os.path.join(path, vehicleFile + '.csv'))

        

if __name__ == "__main__":
    reader = DataReader('data', 'C101', 'lala', '.json')
    reader.save_to_csv('data/solomon_benchmark', 'nodes', 'vehicles')
