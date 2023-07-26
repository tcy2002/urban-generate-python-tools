import numpy as np
import cv2
from numpy.random import random

# 0: Land, 1: Road, 2: Cross, 3: Building, 4: Unavailable
Landmarks: np.ndarray
Parcels = []


def load_data_from_png(path):
    global Landmarks

    img = cv2.imread(path)
    Landmarks = np.zeros((img.shape[1], img.shape[0]), np.uint8)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y, x, 0] == 255:
                Landmarks[x, y] = 1
            elif img[y, x, 1] == 255:
                Landmarks[x, y] = 1
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


def Normalize(Node):
    Magnitude = np.sqrt(Node[0] * Node[0] + Node[1] * Node[1])
    return [Node[0] / Magnitude, Node[1] / Magnitude]


def Get8Neighbors(Node):
    global Landmarks
    x, y = Landmarks.shape
    if 0 < Node[0] < x - 1 and 0 < Node[1] < y - 1:
        return np.array([[Node[0] - 1, Node[1] - 1], [Node[0], Node[1] - 1], [Node[0] + 1, Node[1] - 1],
                         [Node[0] - 1, Node[1]],                             [Node[0] + 1, Node[1]],
                         [Node[0] - 1, Node[1] + 1], [Node[0], Node[1] + 1], [Node[0] + 1, Node[1] + 1]])
    return np.array([])


def Get4Neighbors(Node):
    global Landmarks
    x, y = Landmarks.shape
    if 0 < Node[0] < x - 1 and 0 < Node[1] < y - 1:
        return np.array([[Node[0] - 1, Node[1]], [Node[0], Node[1] + 1], [Node[0] + 1, Node[1]], [Node[0], Node[1] - 1]])
    return np.array([])


def RoadConquestGrowOnOneSide(Index, Node, Radius, OccupiedMap: dict):
    global Landmarks, Parcels

    OccupiedByMe = []
    OccupiedByOther = []

    IRadius = int(Radius)
    while IRadius > 0:
        if Landmarks[Node[0], Node[1]] not in [0, 4]:
            return
        if Landmarks[Node[0], Node[1]] == 0:
            Landmarks[Node[0], Node[1]] = 5
            OccupiedByMe.append(Node)
            Parcels[Index].append(Node)
            OccupiedMap[Node[0], Node[1]] = Index
        else:
            OccupiedByOther.append(Node)
        IRadius -= 1

        Neighbors = Get4Neighbors(Node)
        for Col, Row in Neighbors:
            if Landmarks[Col, Row] in [0, 4]:
                Neighbors8 = Get8Neighbors([Col, Row])
                if len(Neighbors8) == 0:
                    continue
                # exclude the sink point
                if (Landmarks[Neighbors8[1, 0], Neighbors8[1, 1]] == 1 and Landmarks[Neighbors8[6, 0], Neighbors8[6, 1]] == 1) or \
                        (Landmarks[Neighbors8[3, 0], Neighbors8[3, 1]] == 1 and Landmarks[Neighbors8[4, 0], Neighbors8[4, 1]] == 1):
                    continue
                if np.any(Landmarks[Neighbors8[:, 0], Neighbors8[:, 1]] == 1):
                    Node = [Col, Row]
                    break
        else:
            break

    for i in range(len(OccupiedByOther) // 2):
        Parcels[Index].append(OccupiedByOther[i])
        Parcels[OccupiedMap[OccupiedByOther[i][0], OccupiedByOther[i][1]]].remove(OccupiedByOther[i])
        OccupiedMap[OccupiedByOther[i][0], OccupiedByOther[i][1]] = Index
    for n in OccupiedByMe:
        Landmarks[n[0], n[1]] = 4


def RoadConquest(Index, Node, Radius, OccupiedMap: dict):
    global Landmarks, Parcels

    RoadNodes = GetNodesInRadius(Node, Radius, 1)
    if len(RoadNodes) == 0:
        return

    NearestNode = GetNearestNode(Node, RoadNodes)
    Direction = Normalize([Node[0] - NearestNode[0], Node[1] - NearestNode[1]])

    # the nearest node determines that the root node is not on the road
    # and there must be 2 free nodes on both sides of the root node
    RootNode = [round(NearestNode[0] + Direction[0]), round(NearestNode[1] + Direction[1])]
    if Landmarks[RootNode[0], RootNode[1]] != 0:
        return

    Neighbors = Get4Neighbors(RootNode)
    StartNodes = []
    for Col, Row in Neighbors:
        if Landmarks[Col, Row] in [0, 4]:
            Neighbors8 = Get8Neighbors([Col, Row])
            if len(Neighbors8) == 0:
                continue
            if np.any(Landmarks[Neighbors8[:, 0], Neighbors8[:, 1]] == 1):
                StartNodes.append([Col, Row])

    if len(StartNodes) != 2:
        return

    Landmarks[RootNode[0], RootNode[1]] = 5
    OccupiedMap[RootNode[0], RootNode[1]] = Index
    Parcels[Index].append(RootNode)

    RoadConquestGrowOnOneSide(Index, StartNodes[0], Radius, OccupiedMap)
    RoadConquestGrowOnOneSide(Index, StartNodes[1], Radius, OccupiedMap)

    Landmarks[RootNode[0], RootNode[1]] = 4


def GenerateParcels():
    pass


def GenerateOnParcel():
    pass


if __name__ == '__main__':
    load_data_from_png('parcel_test.png')

    img = cv2.imread('parcel_test.png')
    hashmap = dict()

    buildings = np.where(img[:, :, 2] == 255)
    size = len(buildings[0])
    Parcels = [[] for i in range(size)]

    color = [np.uint8([random() * 255, random() * 255, random() * 255]) for i in range(size)]

    for i in range(size):
        RoadConquest(i, [buildings[1][i], buildings[0][i]], 5, hashmap)

    # for key, value in hashmap.items():
    #     img[key[1], key[0]] = color[value]

    for i in range(size):
        for n in Parcels[i]:
            img[n[1], n[0]] = color[i]

    img = cv2.resize(img, (img.shape[1] * 10, img.shape[0] * 10), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)
    cv2.waitKey(0)
