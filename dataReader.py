import pandas as pd
import numpy as np
import sys
import os


class DataReader:
    def __init__(self,
                 dataPath,
                 nodeFile="nodes.csv",
                 vehicleFile="vehicles.csv"
                 ):
        
        # Comprobamos si el directorio especificado en 'dataPath' existe
        if not os.path.exists(dataPath):
            print("No se ha encontrado el path: {}".format(dataPath), file=sys.stderr)
            sys.exit(1)  # Si no existe, salimos del programa

        self.nodePath = os.path.join(dataPath, nodeFile)
        self.vehiclePath = os.path.join(dataPath, vehicleFile)

        # Leer la información de nodos y vehículos
        self.nodeInfo = pd.read_csv(self.nodePath, index_col=0)
        self.vehicleInfo = pd.read_csv(self.vehiclePath, index_col=0)

