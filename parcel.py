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
    if len(Nodes) == 0:
        return MinNode
    for Col, Row in Nodes:
        Distance = (Col - Node[0]) * (Col - Node[0]) + (Row - Node[1]) * (Row - Node[1])
        if Distance < MinDistance:
            MinDistance = Distance
            MinNode = [Col, Row]
    return MinNode


def Magnitude(Node):
    return np.sqrt(Node[0] * Node[0] + Node[1] * Node[1])


def Normalize(Node):
    m = Magnitude(Node)
    return [Node[0] / m, Node[1] / m]


def Distance(Node1, Node2):
    return np.sqrt((Node1[0] - Node2[0]) * (Node1[0] - Node2[0]) +
                   (Node1[1] - Node2[1]) * (Node1[1] - Node2[1]))


def Average(Nodes):
    if len(Nodes) == 0:
        return [0, 0]
    return [sum([x for x, _ in Nodes]) / len(Nodes), sum([y for _, y in Nodes]) / len(Nodes)]


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


def CheckIndex(Node):
    global Landmarks
    return 0 <= Node[0] < Landmarks.shape[0] and 0 <= Node[1] < Landmarks.shape[1]


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
        return [1, 0], 1

    sum_y = sum([y for _, y in Nodes])
    sum_y2 = sum([y * y for _, y in Nodes])
    q = n * sum_y2 - sum_y * sum_y
    if q == 0:
        return [0, 1], 1

    sum_xy = sum([x * y for x, y in Nodes])
    r = n * sum_xy - sum_x * sum_y
    range_x = max([x for x, _ in Nodes]) - min([x for x, _ in Nodes])
    range_y = max([y for _, y in Nodes]) - min([y for _, y in Nodes])
    if range_x > range_y:
        return Normalize([-r/p, 1]), r / p * r / q
    else:
        return Normalize([-1, r/q]), r / p * r / q


def CheckPosition(Root, Node, Direction):
    Diff = [Node[0] - Root[0], Node[1] - Root[1]]
    Factor = Diff[0] * Direction[0] + Diff[1] * Direction[1]
    return abs(Factor) < 3


def RoadConquestGrowOnDirection(Index, Node, Direction, Size, OccupiedMap: dict):
    RootNode = [Node[0], Node[1]]
    DirX = 1 if Direction[0] > 0 else -1
    DirY = 1 if Direction[1] > 0 else -1

    for i in range(0, Size):
        if Direction[0] == 0 or Direction[1] == 0:
            Node = [round(Node[0] + Direction[0]), round(Node[1] + Direction[1])]
        else:
            Node1 = [round(Node[0] + DirX), Node[1]]
            Node2 = [Node[0], round(Node[1] + DirY)]
            Node1Dir = Normalize([Node1[0] - RootNode[0], Node1[1] - RootNode[1]])
            Node2Dir = Normalize([Node2[0] - RootNode[0], Node2[1] - RootNode[1]])
            Node1Cos = Node1Dir[0] * Direction[0] + Node1Dir[1] * Direction[1]
            Node2Cos = Node2Dir[0] * Direction[0] + Node2Dir[1] * Direction[1]
            Node = Node1 if Node1Cos > Node2Cos else Node2
        if not CheckIndex(Node) or Landmarks[Node[0], Node[1]] != 0:
            return
        Landmarks[Node[0], Node[1]] = 4
        Parcels[Index].append(Node)
        OccupiedMap[Node[0], Node[1]] = Index


def RoadConquestGrowOnBothSides(Index, Radius, Direction, OccupiedMap: dict):
    global Landmarks, Parcels

    IRadius = int(Radius)
    TargetSize = 2 * IRadius + 1
    CurrentSize = len(Parcels[Index])
    _, r = LeastSquareLineFitting(Parcels[Index])
    if CurrentSize > TargetSize / 2 and r > 0.5:
        return

    Direction = [-Direction[1], Direction[0]]
    End1, End2 = GetEndNodes(Parcels[Index], Direction)
    CurrentSize = (End2[0] - End1[0]) * Direction[0] + (End2[1] - End1[1]) * Direction[1]
    HalfSize = int((TargetSize - CurrentSize) / 2)

    # Grow on both sides according to the direction
    RoadConquestGrowOnDirection(Index, End2, Direction, HalfSize, OccupiedMap)
    RoadConquestGrowOnDirection(Index, End1, [-Direction[0], -Direction[1]], HalfSize, OccupiedMap)


def RoadConquestGrowOnOneSide(Index, Node, Radius, Direction, OccupiedMap: dict):
    global Landmarks, Parcels

    RootNode = [Node[0], Node[1]]
    OccupiedByOther = []

    IRadius = int(Radius)
    while IRadius > 0:
        Neighbors4 = Get4Neighbors(Node)
        flag = False
        for Col, Row in Neighbors4:
            if Landmarks[Col, Row] in [0, 4]:
                Neighbors8 = Get8Neighbors([Col, Row])
                if len(Neighbors8) > 0 and np.any(Landmarks[Neighbors8[:, 0], Neighbors8[:, 1]] == 1):
                    Node = [Col, Row]
                    flag = True
                    break
        # if the direction deviates too much, stop growing
        if not flag or not CheckPosition(RootNode, Node, Direction):
            break

        if Landmarks[Node[0], Node[1]] == 0:
            Parcels[Index].append(Node)
            OccupiedMap[Node[0], Node[1]] = Index
        else:
            # avoid cutting off other's line
            Neighbors4 = Get4Neighbors(Node)
            if len(Neighbors4) == 0 or np.sum(Landmarks[Neighbors4[:, 0], Neighbors4[:, 1]] == 4) > 1:
                return
            OccupiedByOther.append(Node)
        IRadius -= 1
        Landmarks[Node[0], Node[1]] = 5

    # use fairness principle to divide the conflict region
    Size = len(OccupiedByOther)
    for i in range(Size):
        if i < Size / 2:
            Parcels[Index].append(OccupiedByOther[i])
            Parcels[OccupiedMap[OccupiedByOther[i][0], OccupiedByOther[i][1]]].remove(OccupiedByOther[i])
            OccupiedMap[OccupiedByOther[i][0], OccupiedByOther[i][1]] = Index
        else:
            Landmarks[OccupiedByOther[i][0], OccupiedByOther[i][1]] = 4


def RoadConquest(Index, Node, Radius, OccupiedMap: dict):
    global Landmarks, Parcels

    RoadNodes = GetNodesInRadius(Node, Radius, 1)
    if len(RoadNodes) == 0:
        return

    NearestNode = GetNearestNode(Node, RoadNodes)
    Direction = Normalize([Node[0] - NearestNode[0], Node[1] - NearestNode[1]])
    Abs_Dir = [abs(Direction[0]), abs(Direction[1])]
    Radius *= np.sqrt(1 + np.square(min(Abs_Dir[0], Abs_Dir[1]) / max(Abs_Dir[0], Abs_Dir[1])))

    # the nearest node determines that the root node is not on the road
    # and there must be 2 free nodes on both sides of the root node
    RootNode = [round(NearestNode[0] + Direction[0]), round(NearestNode[1] + Direction[1])]
    if Landmarks[RootNode[0], RootNode[1]] != 0:
        return

    Landmarks[RootNode[0], RootNode[1]] = 5
    OccupiedMap[RootNode[0], RootNode[1]] = Index
    Parcels[Index].append(RootNode)

    # grow on both sides according to radius
    RoadConquestGrowOnOneSide(Index, RootNode, Radius, Direction, OccupiedMap)
    RoadConquestGrowOnOneSide(Index, RootNode, Radius, Direction, OccupiedMap)

    for Col, Row in Parcels[Index]:
        Landmarks[Col, Row] = 4

    # grow on both sides if not enough
    RoadConquestGrowOnBothSides(Index, Radius, Direction, OccupiedMap)


def RegionConquestCheck(Index, OccupiedMap: dict):
    IndicesToRemove = []
    for i, Node in enumerate(Parcels[Index]):
        Neighbors4 = Get4Neighbors(Node)
        flag = False
        for Col, Row in Neighbors4:
            if Landmarks[Col, Row] == 4 and OccupiedMap[Col, Row] == Index:
                flag = True
                break
        if not flag:
            IndicesToRemove.append(i)

    for i in reversed(IndicesToRemove):
        Landmarks[Parcels[Index][i][0], Parcels[Index][i][1]] = 0
        OccupiedMap.pop((Parcels[Index][i][0], Parcels[Index][i][1]))
        Parcels[Index].pop(i)


def RegionConquestGrowOnOneNode(Index, Node, Size, Direction, OccupiedMap: dict):
    global Landmarks, Parcels

    for i in range(1, Size):
        NewNode = [round(Node[0] + Direction[0] * i), round(Node[1] + Direction[1] * i)]
        if not CheckIndex(NewNode):
            break

        if Landmarks[NewNode[0], NewNode[1]] == 0:
            Parcels[Index].append(NewNode)
            OccupiedMap[NewNode[0], NewNode[1]] = Index
            Landmarks[NewNode[0], NewNode[1]] = 5
        elif Landmarks[NewNode[0], NewNode[1]] == 4:
            if OccupiedMap[NewNode[0], NewNode[1]] != Index:
                break
            else:
                continue
        elif Landmarks[NewNode[0], NewNode[1]] in [5, 3]:
            continue
        else:
            break

        # avoid crossing the road
        Neighbors4 = Get4Neighbors(NewNode)
        if len(Neighbors4) > 0 and np.sum(Landmarks[Neighbors4[:, 0], Neighbors4[:, 1]] == 1) > 1:
            break


def RegionConquest(Index, Radius, OccupiedMap: dict):
    global Landmarks, Parcels

    if len(Parcels[Index]) == 0:
        return

    # check if there are nodes that are not adjacent to the region
    RegionConquestCheck(Index, OccupiedMap)

    Direction, _ = LeastSquareLineFitting(Parcels[Index])
    TargetSize = Radius * 2 + 1
    Center = Average(Parcels[Index])

    # check if the direction should be reversed
    check = np.array([[round(Col + Direction[0]), round(Row + Direction[1])] for Col, Row in Parcels[Index]])
    if np.any(Landmarks[check[:, 0], check[:, 1]] == 1):
        Direction = [-Direction[0], -Direction[1]]

    TargetSizeForNodes = [round(TargetSize + ((Center[0] - Col) * Direction[0] + (Center[1] - Row) * Direction[1]))
                          for Col, Row in Parcels[Index]]

    ParcelsCopy = [n for n in Parcels[Index]]
    for i, Node in enumerate(ParcelsCopy):
        RegionConquestGrowOnOneNode(Index, Node, TargetSizeForNodes[i], Direction, OccupiedMap)

    for Col, Row in Parcels[Index]:
        Landmarks[Col, Row] = 4


def GenerateParcels():
    pass


def GenerateOnParcel():
    pass


if __name__ == '__main__':
    file = 'parcel.png'
    load_data_from_png(file)
    img = cv2.imread(file)
    hashmap = dict()

    buildings = np.where(img[:, :, 2] == 255)
    size = len(buildings[0])
    Parcels = [[] for i in range(size)]

    color = [np.uint8([random() * 255, random() * 255, random() * 255]) for i in range(size)]

    radius = 15
    target = range(size)
    # target = [25]
    for i, x in enumerate(target):
        RoadConquest(i, [buildings[1][x], buildings[0][x]], radius, hashmap)
    for i, x in enumerate(target):
        RegionConquest(i, radius, hashmap)

    for key, value in hashmap.items():
        img[key[1], key[0]] = color[value]

    # for i in range(size):
    #     if len(Parcels[i]) == 0:
    #         continue
    #     for p in Parcels[i]:
    #         img[p[1], p[0]] = color[i]

    # mask = np.where(Landmarks == 5)
    # img[mask[1], mask[0]] = [255, 255, 255]

    img = cv2.resize(img, (img.shape[1] * 5, img.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)
    cv2.waitKey(0)
