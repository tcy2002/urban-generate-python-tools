import numpy as np
import cv2
from numpy.random import random


# 0: Land, 1: Road, 3x: Building, 4:Occupied
Landmarks: np.ndarray
Buildings = []


def load_data_from_png(path):
    global Landmarks

    img = cv2.imread(path)
    Landmarks = np.zeros((img.shape[1], img.shape[0]), np.uint8)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y, x, 0] == 255 and img[y, x, 1] == 0 and img[y, x, 2] == 0:
                Landmarks[x, y] = 1
            elif img[y, x, 0] == 0 and img[y, x, 1] == 0 and img[y, x, 2] == 255:
                Landmarks[x, y] = 3
            # if img[y, x, 0] == 0 and img[y, x, 1] == 255 and img[y, x, 2] == 0:
            #     Landmarks[x, y] = 31
            # elif img[y, x, 0] == 255 and img[y, x, 1] == 0 and img[y, x, 2] == 0:
            #     Landmarks[x, y] = 32
            # elif img[y, x, 0] == 0 and img[y, x, 1] == 0 and img[y, x, 2] == 255:
            #     Landmarks[x, y] = 32
            # elif img[y, x, 0] == 255 and img[y, x, 1] == 255 and img[y, x, 2] == 255:
            #     Landmarks[x, y] = 1


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
        D = Distance(Node, [Col, Row])
        if D < MinDistance:
            MinDistance = D
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
            [[Node[0] - 1, Node[1]], [Node[0] + 1, Node[1]], [Node[0], Node[1] - 1], [Node[0], Node[1] + 1]])
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


class Building:
    def __init__(self, Index, Radius, Center):
        self.Index = Index
        self.Radius = Radius
        self.Center = Center
        self.Nodes = []

        self.__Direction = [0, 1]

        self.__RoadRootNode = [-1, -1]
        self.__RoadEndNodes = [[-1, -1], [-1, -1]]
        self.__RoadConqueredDist = [0, 0]
        self.__RoadTerminalDist = 0
        self.__RoadEndFlag = [False, False]

        self.__RegionStartNum = 0
        self.__RegionDirection = [0, 1]
        self.__RegionConqueredDist = []
        self.__RegionTerminalDist = []
        self.__RegionEndFlag = []

    def RoadConquestEnds(self):
        return self.__RoadEndFlag[0] and self.__RoadEndFlag[1]

    def __RoadConquestCheckDirection(self, Node):
        Diff = [Node[0] - self.__RoadRootNode[0], Node[1] - self.__RoadRootNode[1]]
        Factor = Diff[0] * self.__Direction[0] + Diff[1] * self.__Direction[1]
        return abs(Factor) < 2.5

    def __RoadConquestCheckOcclusion(self, Node, OccupiedMap: dict):
        global Landmarks

        for i in range(1, int(self.__RoadTerminalDist) // 2):
            CheckNode = [round(Node[0] + self.__Direction[0] * i),
                         round(Node[1] + self.__Direction[1] * i)]
            if CheckIndex(CheckNode) and Landmarks[CheckNode[0], CheckNode[1]] == 4 \
                    and OccupiedMap[CheckNode[0], CheckNode[1]] != self.Index:
                return True
        return False

    def RoadConquestInit(self, OccupiedMap: dict):
        global Landmarks

        RoadNodes = GetNodesInRadius(self.Center, self.Radius, 1)
        if len(RoadNodes) == 0:
            return True

        NearestNode = GetNearestNode(self.Center, RoadNodes)
        self.__Direction = Normalize([self.Center[0] - NearestNode[0], self.Center[1] - NearestNode[1]])
        Abs_Dir = [abs(self.__Direction[0]), abs(self.__Direction[1])]
        self.__RoadTerminalDist = self.Radius * np.sqrt(1 + np.square(min(Abs_Dir[0], Abs_Dir[1])
                                                                      / max(Abs_Dir[0], Abs_Dir[1])))
        self.__RoadRootNode = [round(NearestNode[0] + self.__Direction[0]), round(NearestNode[1] + self.__Direction[1])]

        if Landmarks[self.__RoadRootNode[0], self.__RoadRootNode[1]] != 0:
            return True

        self.__RoadEndNodes[0] = self.__RoadEndNodes[1] = self.__RoadRootNode
        self.Nodes.append(self.__RoadRootNode)
        Landmarks[self.__RoadRootNode[0], self.__RoadRootNode[1]] = 4
        OccupiedMap[self.__RoadRootNode[0], self.__RoadRootNode[1]] = self.Index
        return False

    def __RoadConquestStep1(self, EndNodeIndex,  OccupiedMap: dict):
        global Landmarks

        # grow along the road
        CurrentNode = self.__RoadEndNodes[EndNodeIndex]
        Neighbors4 = Get4Neighbors(CurrentNode)
        Node = [-1, -1]
        for Neighbor in Neighbors4:
            if Landmarks[Neighbor[0], Neighbor[1]] == 0:
                Neighbors8 = Get8Neighbors(Neighbor)
                if len(Neighbors8) > 0 and np.any(Landmarks[Neighbors8[:, 0], Neighbors8[:, 1]] == 1):
                    Node = Neighbor
                    break
        if Node[0] == -1 or not self.__RoadConquestCheckDirection(Node) \
                or self.__RoadConquestCheckOcclusion(Node, OccupiedMap):
            return True

        self.Nodes.append(Node)
        Landmarks[Node[0], Node[1]] = 4
        OccupiedMap[Node[0], Node[1]] = self.Index
        self.__RoadEndNodes[EndNodeIndex] = Node
        self.__RoadConqueredDist[EndNodeIndex] += 1

        if self.__RoadConqueredDist[EndNodeIndex] < int(self.__RoadTerminalDist):
            return False
        return True

    def RoadConquestStepN(self, N, OccupiedMap: dict):
        global Landmarks

        if len(self.Nodes) == 0:
            return

        for _ in range(N):
            if not self.__RoadEndFlag[0]:
                self.__RoadEndFlag[0] = self.__RoadConquestStep1(0, OccupiedMap)
            if not self.__RoadEndFlag[1]:
                self.__RoadEndFlag[1] = self.__RoadConquestStep1(1, OccupiedMap)
            if self.__RoadEndFlag[0] and self.__RoadEndFlag[1]:
                break

    def RegionConquestEnds(self):
        return np.all(self.__RegionEndFlag)

    def RegionConquestInit(self):
        global Landmarks

        self.__RegionEndFlag = [False for _ in self.Nodes]
        self.__RegionConqueredDist = [0 for _ in self.Nodes]
        self.__Direction, _ = LeastSquareLineFitting(self.Nodes)
        self.__RegionStartNum = len(self.Nodes)

        # check if the direction should be reversed
        Check = np.array([[round(Col + self.__Direction[0]), round(Row + self.__Direction[1])]
                          for Col, Row in self.Nodes])
        if np.any(Landmarks[Check[:, 0], Check[:, 1]] == 1):
            self.__Direction = [-self.__Direction[0], -self.__Direction[1]]

        Center = Average(self.Nodes)
        self.__RegionTerminalDist = [round(self.Radius * 2 + ((Center[0] - Col) * self.__Direction[0] +
                                                              (Center[1] - Row) * self.__Direction[1]))
                                     for Col, Row in self.Nodes]

        return False

    def __RegionConquestStep1(self, EndNodeIndex, OccupiedMap: dict):
        global Landmarks, img, color

        Length = self.__RegionConqueredDist[EndNodeIndex] + 1
        Node = [round(self.Nodes[EndNodeIndex][0] + self.__Direction[0] * Length),
                round(self.Nodes[EndNodeIndex][1] + self.__Direction[1] * Length)]
        if not CheckIndex(Node):
            return True

        if Landmarks[Node[0], Node[1]] == 0:
            self.Nodes.append(Node)
            Landmarks[Node[0], Node[1]] = 4
            OccupiedMap[Node[0], Node[1]] = self.Index
            self.__RegionConqueredDist[EndNodeIndex] += 1
            img[Node[1], Node[0]] = color[self.__RegionConqueredDist[EndNodeIndex]]

            # avoid crossing the road
            Neighbors4 = Get4Neighbors(Node)
            if len(Neighbors4) > 0 and np.sum(Landmarks[Neighbors4[:, 0], Neighbors4[:, 1]] == 1) > 1:
                return True
        elif Landmarks[Node[0], Node[1]] == 3 or \
                (Landmarks[Node[0], Node[1]] == 4 and OccupiedMap[Node[0], Node[1]] == self.Index):
            self.__RegionConqueredDist[EndNodeIndex] += 1
        else:
            return True

        if self.__RegionConqueredDist[EndNodeIndex] < self.__RegionTerminalDist[EndNodeIndex]:
            return False
        return True

    def RegionConquestStepN(self, N, OccupiedMap: dict):
        global Landmarks

        if len(self.Nodes) == 0:
            return

        for _ in range(N):
            for i in range(self.__RegionStartNum):
                if not self.__RegionEndFlag[i]:
                    self.__RegionEndFlag[i] = self.__RegionConquestStep1(i, OccupiedMap)
            if np.all(self.__RegionEndFlag):
                break


def RoadConquest(OccupiedMap: dict):
    global Landmarks

    EndFlag = [False for _ in Buildings]
    for Building in Buildings:
        EndFlag[Building.Index] = Building.RoadConquestInit(OccupiedMap)

    while True:
        for Building in Buildings:
            if not EndFlag[Building.Index]:
                Building.RoadConquestStepN(1, OccupiedMap)
                EndFlag[Building.Index] = Building.RoadConquestEnds()
        if np.all(EndFlag):
            break


def RegionConquest(OccupiedMap: dict):
    global Landmarks, img, color

    EndFlag = [False for _ in Buildings]
    for Building in Buildings:
        EndFlag[Building.Index] = Building.RegionConquestInit()

    # for Key, Value in hashmap.items():
    #     img[Key[1], Key[0]] = color[Value]
    # cv2.imshow('img', cv2.resize(img, (img.shape[1] * 5, img.shape[0] * 5), interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)

    while True:
        for Building in Buildings:
            if not EndFlag[Building.Index]:
                Building.RegionConquestStepN(1, OccupiedMap)
                EndFlag[Building.Index] = Building.RegionConquestEnds()
                # cv2.imshow('img', cv2.resize(img, (img.shape[1] * 5, img.shape[0] * 5), interpolation=cv2.INTER_NEAREST))
                # cv2.waitKey(0)
        if np.all(EndFlag):
            break


if __name__ == '__main__':
    file = 'parcel.png'
    load_data_from_png(file)
    img = cv2.imread(file)
    hashmap = dict()

    buildings = np.where(Landmarks == 3)
    size = len(buildings[0])
    color = [np.uint8([random() * 255, random() * 255, random() * 255]) for i in range(size)]

    radius = 15
    target = range(size)
    # target = [48]
    for i, x in enumerate(target):
        Buildings.append(Building(i, radius, [buildings[0][x], buildings[1][x]]))

    RoadConquest(hashmap)
    RegionConquest(hashmap)

    for Key, Value in hashmap.items():
        img[Key[1], Key[0]] = color[Value]

    # for Building in Buildings:
    #     for Node in Building.Nodes:
    #         img[Node[1], Node[0]] = color[Building.Index]

    img = cv2.resize(img, (img.shape[1] * 5, img.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)
    cv2.waitKey(0)
