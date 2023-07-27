import numpy as np
import cv2
from numpy.random import random

# 0: Land, 1: Road, 2: Cross, 3: Building, 4: Unavailable
Landmarks: np.ndarray
Parcels = []
AngleRef = [[8, 6, 2], [0, 7, 1], [4, 5, 3]]


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
    if len(Nodes) == 0:
        return MinNode
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
                         [Node[0] - 1, Node[1]], [Node[0] + 1, Node[1]],
                         [Node[0] - 1, Node[1] + 1], [Node[0], Node[1] + 1], [Node[0] + 1, Node[1] + 1]])
    return np.array([])


def Get4Neighbors(Node):
    global Landmarks
    x, y = Landmarks.shape
    if 0 < Node[0] < x - 1 and 0 < Node[1] < y - 1:
        return np.array(
            [[Node[0] - 1, Node[1]], [Node[0], Node[1] + 1], [Node[0] + 1, Node[1]], [Node[0], Node[1] - 1]])
    return np.array([])


def GetAngle(Node1, Node2):
    global AngleRef

    Dir = Normalize([Node2[0] - Node1[0], Node2[1] - Node1[1]])
    Dir = [round(Dir[0]), round(Dir[1])]
    return AngleRef[Dir[0]][Dir[1]]


def CheckAngle(AngleA, AngleB, AngleT):
    Side1 = [(AngleT + 1) % 8, (AngleT + 2) % 8, (AngleT + 3) % 8]
    Side2 = [(AngleT + 5) % 8, (AngleT + 6) % 8, (AngleT + 7) % 8]
    return (AngleA in Side1 and AngleB in Side1) or (AngleA in Side2 and AngleB in Side2)


def CheckIndex(Node):
    global Landmarks
    return 0 <= Node[0] < Landmarks.shape[0] and 0 <= Node[1] < Landmarks.shape[1]


def Average(Nodes):
    if len(Nodes) == 0:
        return [0, 0]
    return [sum([x for x, _ in Nodes]) / len(Nodes), sum([y for _, y in Nodes]) / len(Nodes)]


def GetEndNodes(Nodes, Direction):
    Center = Average(Nodes)
    Result = []
    for Node in Nodes:
        diff = [Node[0] - Center[0], Node[1] - Center[1]]
        factor = diff[0] * Direction[0] + diff[1] * Direction[1]
        Result.append([Node, factor])
    Result.sort(key=lambda x: x[1])
    return Result[0][0], Result[-1][0]


def LeastSquareLineFitting(Nodes):
    n = len(Nodes)

    sum_x = sum([x for x, _ in Nodes])
    sum_x2 = sum([x * x for x, _ in Nodes])
    p = n * sum_x2 - sum_x * sum_x
    if p == 0:
        return [1, 0]

    sum_y = sum([y for _, y in Nodes])
    sum_y2 = sum([y * y for _, y in Nodes])
    q = n * sum_y2 - sum_y * sum_y
    if q == 0:
        return [0, 1]

    sum_xy = sum([x * y for x, y in Nodes])
    r = n * sum_xy - sum_x * sum_y
    range_x = max([x for x, _ in Nodes]) - min([x for x, _ in Nodes])
    range_y = max([y for _, y in Nodes]) - min([y for _, y in Nodes])
    if range_x > range_y:
        return Normalize([-r, p])
    else:
        return Normalize([-q, r])


def DivideConflict(Index, Occupied, NonOccupied, OccupiedMap: dict):
    Size = len(Occupied)
    for i in range(Size):
        if i < Size // 2:
            Parcels[Index].append(Occupied[i])
            Parcels[OccupiedMap[Occupied[i][0], Occupied[i][1]]].remove(Occupied[i])
            OccupiedMap[Occupied[i][0], Occupied[i][1]] = Index
        else:
            NonOccupied.append(Occupied[i])


def RoadConquestGrowOnBothSides(Index, Radius, OccupiedMap: dict):
    global Landmarks, Parcels

    IRadius = int(Radius)
    TargetSize = 2 * IRadius + 1
    CurrentSize = len(Parcels[Index])

    if CurrentSize > TargetSize / 2:
        return

    HalfSize = (TargetSize - CurrentSize) // 2
    Normal = LeastSquareLineFitting(Parcels[Index])
    Direction = [-Normal[1], Normal[0]]

    End1, End2 = GetEndNodes(Parcels[Index], Direction)
    for i in range(HalfSize):
        Node1 = [round(End1[0] - Direction[0] * (i + 1)), round(End1[1] - Direction[1] * (i + 1))]
        Node2 = [round(End2[0] + Direction[0] * (i + 1)), round(End2[1] + Direction[1] * (i + 1))]
        if Landmarks[Node1[0], Node1[1]] == 0:
            Landmarks[Node1[0], Node1[1]] = 4
            Parcels[Index].append(Node1)
            OccupiedMap[Node1[0], Node1[1]] = Index
        if Landmarks[Node2[0], Node2[1]] == 0:
            Landmarks[Node2[0], Node2[1]] = 4
            Parcels[Index].append(Node2)
            OccupiedMap[Node2[0], Node2[1]] = Index


def RoadConquestGrowOnOneSide(Index, Node, Radius, Angle, OccupiedMap: dict, NonOccupied: list):
    global Landmarks, Parcels

    RootNode = OldNode = [Node[0], Node[1]]
    OccupiedByOther = []

    IRadius = int(Radius)
    while IRadius > 0:
        Neighbors4 = Get4Neighbors(Node)
        flag = False
        for Col, Row in Neighbors4:
            if Landmarks[Col, Row] in [0, 4]:
                Neighbors8 = Get8Neighbors([Col, Row])
                if len(Neighbors8) == 0:
                    continue
                if np.any(Landmarks[Neighbors8[:, 0], Neighbors8[:, 1]] == 1):
                    Node = [Col, Row]
                    flag = True
                    break
        if not flag:
            break

        Neighbors4 = Get4Neighbors(Node)
        if np.any(Landmarks[Neighbors4[:, 0], Neighbors4[:, 1]] == 1):
            if not CheckAngle(GetAngle(RootNode, Node), GetAngle(OldNode, Node), Angle):
                break
            if Landmarks[Node[0], Node[1]] == 0:
                Parcels[Index].append(Node)
                OccupiedMap[Node[0], Node[1]] = Index
            else:
                OccupiedByOther.append(Node)
            IRadius -= 1
            OldNode = [Node[0], Node[1]]
        else:
            NonOccupied.append(Node)
        Landmarks[Node[0], Node[1]] = 5

    # use fairness principle to divide the conflict region
    DivideConflict(Index, OccupiedByOther, NonOccupied, OccupiedMap)


def RoadConquest(Index, Node, Radius, OccupiedMap: dict):
    global Landmarks, Parcels

    RoadNodes = GetNodesInRadius(Node, Radius, 1)
    if len(RoadNodes) == 0:
        return

    NearestNode = GetNearestNode(Node, RoadNodes)
    Direction = Normalize([Node[0] - NearestNode[0], Node[1] - NearestNode[1]])

    # the nearest node determines that the root node is not on the road
    # and there must be 2 free nodes on both sides of the root node
    RootNode = [round(NearestNode[0] + Direction[0]), NearestNode[1]] if abs(Direction[0]) > abs(
        Direction[1]) else [NearestNode[0], round(NearestNode[1] + Direction[1])]
    if Landmarks[RootNode[0], RootNode[1]] != 0:
        return

    Landmarks[RootNode[0], RootNode[1]] = 5
    OccupiedMap[RootNode[0], RootNode[1]] = Index
    Parcels[Index].append(RootNode)

    Dir = GetAngle(NearestNode, RootNode)
    NonOccupied = []

    # grow on both sides according to radius
    RoadConquestGrowOnOneSide(Index, RootNode, Radius, Dir, OccupiedMap, NonOccupied)
    RoadConquestGrowOnOneSide(Index, RootNode, Radius, Dir, OccupiedMap, NonOccupied)

    for Col, Row in Parcels[Index]:
        Landmarks[Col, Row] = 4
    for Col, Row in NonOccupied:
        Landmarks[Col, Row] = 4 if (Col, Row) in OccupiedMap else 0

    # grow on both sides if not enough
    RoadConquestGrowOnBothSides(Index, Radius, OccupiedMap)


def RegionConquestGrowOnOnePoint(Index, Node, Size, Direction, OccupiedMap: dict, NonOccupied: list):
    global Landmarks, Parcels

    OccupiedByOther = []

    for i in range(Size):
        NewNode = [round(Node[0] + Direction[0] * i), round(Node[1] + Direction[1] * i)]
        if not CheckIndex(NewNode):
            break
        if Landmarks[NewNode[0], NewNode[1]] in [0, 3]:
            Parcels[Index].append(NewNode)
            OccupiedMap[NewNode[0], NewNode[1]] = Index
        elif Landmarks[NewNode[0], NewNode[1]] == 4:
            OccupiedByOther.append(NewNode)
        elif Landmarks[NewNode[0], NewNode[1]] == 5:
            continue
        else:
            break
        Landmarks[NewNode[0], NewNode[1]] = 5

    # also use fairness principle to conquest the conflict region
    DivideConflict(Index, OccupiedByOther, NonOccupied, OccupiedMap)


def RegionConquestFill(Index, OccupiedMap: dict):
    global Landmarks, Parcels

    if len(Parcels[Index]) == 0:
        return

    ParcelsCopy = [n for n in Parcels[Index]]
    for Col, Row in ParcelsCopy:
        Neighbors4 = Get4Neighbors([Col, Row])
        for M, N in Neighbors4:
            if Landmarks[M, N] in [0, 3]:
                Neighbors4N = Get4Neighbors([M, N])
                if len(Neighbors4N) == 0:
                    continue
                if np.sum(Landmarks[Neighbors4N[:, 0], Neighbors4N[:, 1]] == 4) > 2:
                    Landmarks[M, N] = 4
                    Parcels[Index].append([M, N])
                    OccupiedMap[M, N] = Index


def RegionConquest(Index, Radius, OccupiedMap: dict):
    global Landmarks, Parcels

    if len(Parcels[Index]) == 0:
        return

    Direction = LeastSquareLineFitting(Parcels[Index])
    TargetSize = int(Radius * 2 + 1)

    # check if the direction should be reversed
    check = np.array([[round(Col + Direction[0]), round(Row + Direction[1])] for Col, Row in Parcels[Index]])
    if np.any(Landmarks[check[:, 0], check[:, 1]] == 1):
        Direction = [-Direction[0], -Direction[1]]

    NonOccupied = []
    ParcelsCopy = [n for n in Parcels[Index]]
    for Node in ParcelsCopy:
        RegionConquestGrowOnOnePoint(Index, Node, TargetSize, Direction, OccupiedMap, NonOccupied)

    for Col, Row in Parcels[Index]:
        Landmarks[Col, Row] = 4
    for Col, Row in NonOccupied:
        Landmarks[Col, Row] = 4 if (Col, Row) in OccupiedMap else 0

    # fill the empty points
    RegionConquestFill(Index, OccupiedMap)


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
    for i in range(size):
        RegionConquest(i, 5, hashmap)

    for key, value in hashmap.items():
        img[key[1], key[0]] = color[value]

    # for i in range(size):
    #     if len(Parcels[i]) == 0:
    #         continue
    #     for p in Parcels[i]:
    #         img[p[1], p[0]] = color[i]

    # mask = np.where(Landmarks == 4)
    # img[mask[1], mask[0]] = [0, 0, 255]

    img = cv2.resize(img, (img.shape[1] * 10, img.shape[0] * 10), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    # a = [[4, 15], [3, 16], [2, 17], [1, 18], [5, 15], [6, 14], [7, 13], [8, 13], [9, 13]]
    # b = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 4], [6, 5], [7, 6]]
    # c = [[1, 6], [2, 5], [3, 4], [4, 3], [5, 3], [6, 2], [7, 1]]
    d = [[1, 1], [2, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 7]]
    # print(LeastSquareLineFitting(a))
    # print(LeastSquareLineFitting(b))
    # print(LeastSquareLineFitting(c))
    print(LeastSquareLineFitting(d))
