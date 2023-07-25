import numpy as np
import cv2


class HashMap:
    def __init__(self):
        self.Map = dict()

    def __getitem__(self, key):
        return self.Map[key]

    def __setitem__(self, key, value):
        self.Map[key] = value

    def __contains__(self, key):
        return key in self.Map

    def __len__(self):
        return len(self.Map)

    def __iter__(self):
        return iter(self.Map)

    def __delitem__(self, key):
        del self.Map[key]


class FBuilding:
    def __init__(self):
        self.Nodes = [0, 0]
        self.Type = 0
        self.Radius = 5
        self.ConnectedToRoad = False
        self.OccupiedNodes = []


class FParcel:
    def __init__(self):
        self.Source = 0
        self.Direction = [1, 0]
        self.ForwardNodes = []


# 0: Land, 1: Road, 2: Cross, 3: Building
Landmarks: np.ndarray
Buildings = []
Parcels = []


def load_data_from_png(path):
    global Landmarks, Buildings

    img = cv2.imread(path)
    Landmarks = np.zeros((img.shape[1], img.shape[0]), np.uint8)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y, x, 0] == 255:
                Landmarks[x, y] = 1
            elif img[y, x, 1] == 255:
                Landmarks[x, y] = 2
            elif img[y, x, 2] == 255:
                Landmarks[x, y] = 3


def GetNodesInRadius(Node, Radius, Type):
    Result = []
    DRadius = Radius + 0.3
    IRadius = int(Radius)
    CentralCol, CentralRow = Node

    for Row in range(CentralRow - IRadius if CentralRow - IRadius > 0 else 0,
                     CentralRow + IRadius + 1 if CentralRow + IRadius < Landmarks.shape[1] else Landmarks.shape[1]):
        RowHalfLength = int(np.sqrt(DRadius * DRadius - (Row - CentralRow) * (Row - CentralRow)))
        for Col in range(CentralCol - RowHalfLength if CentralCol - RowHalfLength > 0 else 0,
                         CentralCol + RowHalfLength + 1 if CentralCol + RowHalfLength < Landmarks.shape[0] else
                         Landmarks.shape[0]):
            if Landmarks[Col, Row] == Type:
                Result.append([Col, Row])

    return Result


def GetNearestNode(Node, Nodes):
    MinDistance = 100000.0
    MinNode = [-1, -1]
    for Col, Row in Nodes:
        Distance = (Col - Node[0]) * (Col - Node[0]) + (Row - Node[1]) * (Row - Node[1])
        if Distance < MinDistance:
            MinDistance = Distance
            MinNode = [Col, Row]
    return MinNode


def GenerateParcels():
    pass


def GenerateOnParcel(Node, Radius, Source, OccupiedMap: HashMap):
    global Landmarks, Buildings, Parcels

    RoadNodes = GetNodesInRadius(Node, Radius, 1)
    NearestNode = GetNearestNode(Node, RoadNodes)

    

    return RoadNodes, NearestNode


if __name__ == '__main__':
    load_data_from_png('parcel_test.png')

    img = cv2.imread('parcel_test.png')
    center = [30, 21]
    nodes, nearest_node = GenerateOnParcel(center, 5, 0, HashMap())
    for node in nodes:
        img[node[1], node[0]] = [255, 255, 255]
    img[nearest_node[1], nearest_node[0]] = [0, 255, 255]
    img[center[1], center[0]] = [255, 255, 0]
    img = cv2.resize(img, (img.shape[1] * 10, img.shape[0] * 10), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)
    cv2.waitKey(0)


