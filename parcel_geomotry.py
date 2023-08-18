import numpy as np
import cv2
from numpy.random import random

# algorithm of parcel division based on geometry

# 0: Land, 1: Road, 1x: adjacent to road, 3x: Building, 4:Occupied by building
Landmarks: np.ndarray
Buildings = []


def load_data_from_png(path):
    global Landmarks

    img = cv2.imread(path)
    Landmarks = np.zeros((img.shape[1], img.shape[0]), np.uint8)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y, x, 0] == 255 and img[y, x, 1] == 255 and img[y, x, 2] == 255:
                Landmarks[x, y] = 1
            elif img[y, x, 0] == 0 and img[y, x, 1] == 255 and img[y, x, 2] == 255:
                Landmarks[x, y] = 3


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


def DistanceTransformForRoad(img):
    global Landmarks

    Marks = [1, 11, 12, 13, 14]
    Colors = [[255, 255, 255], [140, 140, 140], [110, 110, 110], [80, 80, 80], [50, 50, 50]]
    for n in range(4):
        Nodes = np.where(Landmarks == Marks[n])
        for i in range(len(Nodes[0])):
            x = Nodes[0][i]
            y = Nodes[1][i]
            if 0 < x < Landmarks.shape[0] - 1 and 0 < y < Landmarks.shape[1] - 1:
                if Landmarks[x - 1, y] == 0:
                    Landmarks[x - 1, y] = Marks[n + 1]
                    img[y, x - 1, :] = Colors[n + 1]
                if Landmarks[x + 1, y] == 0:
                    Landmarks[x + 1, y] = Marks[n + 1]
                    img[y, x + 1, :] = Colors[n + 1]
                if Landmarks[x, y - 1] == 0:
                    Landmarks[x, y - 1] = Marks[n + 1]
                    img[y - 1, x, :] = Colors[n + 1]
                if Landmarks[x, y + 1] == 0:
                    Landmarks[x, y + 1] = Marks[n + 1]
                    img[y + 1, x, :] = Colors[n + 1]


def Distance(Node1, Node2):
    return np.sqrt((Node1[0] - Node2[0]) * (Node1[0] - Node2[0]) +
                   (Node1[1] - Node2[1]) * (Node1[1] - Node2[1]))


def Average(Nodes):
    if len(Nodes) == 0:
        return [0, 0]
    return [sum([x for x, _ in Nodes]) / len(Nodes), sum([y for _, y in Nodes]) / len(Nodes)]


def Magnitude(Node):
    return np.sqrt(Node[0] * Node[0] + Node[1] * Node[1])


def Normalize(Node):
    m = Magnitude(Node)
    return [Node[0] / m, Node[1] / m]


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


def GetNodesInRadius(Node, Radius, Type):
    Result = []
    DRadius = Radius + 0.3
    IRadius = int(Radius)
    CentralCol = round(Node[0])
    CentralRow = round(Node[1])
    for Row in range(CentralRow - IRadius if CentralRow - IRadius > 0 else 0,
                     CentralRow + IRadius + 1 if CentralRow + IRadius < Landmarks.shape[1] else Landmarks.shape[1]):
        RowHalfLength = int(np.sqrt(DRadius * DRadius - (Row - CentralRow) * (Row - CentralRow)))
        for Col in range(CentralCol - RowHalfLength if CentralCol - RowHalfLength > 0 else 0,
                         CentralCol + RowHalfLength + 1 if CentralCol + RowHalfLength < Landmarks.shape[0] else
                         Landmarks.shape[0]):
            if Landmarks[Col, Row] == Type:
                Result.append([Col, Row])
    return Result


def GetNextIntersectNode(CurrentNode, RootPos, Direction):
    if Direction[0] == 0:
        return [CurrentNode[0], CurrentNode[1] + (1 if Direction[1] > 0 else -1)], \
            [RootPos[0], CurrentNode[1] + (0.5 if Direction[1] > 0 else -0.5)]
    elif Direction[1] == 0:
        return [CurrentNode[0] + (1 if Direction[0] > 0 else -1), CurrentNode[1]], \
            [CurrentNode[0] + (0.5 if Direction[0] > 0 else -0.5), RootPos[1]]
    else:
        th = (CurrentNode[1] - RootPos[1] + (0.5 if Direction[1] > 0 else -0.5)) / Direction[1]
        tv = (CurrentNode[0] - RootPos[0] + (0.5 if Direction[0] > 0 else -0.5)) / Direction[0]
        if th < tv:
            return [CurrentNode[0], CurrentNode[1] + (1 if Direction[1] > 0 else -1)], \
                [RootPos[0] + Direction[0] * th, RootPos[1] + Direction[1] * th]
        else:
            return [CurrentNode[0] + (1 if Direction[0] > 0 else -1), CurrentNode[1]], \
                [RootPos[0] + Direction[0] * tv, RootPos[1] + Direction[1] * tv]


def GetIntersectNodes(RootPos, EndPos):
    RootNode = [round(RootPos[0]), round(RootPos[1])]
    EndNode = [round(EndPos[0]), round(EndPos[1])]
    Direction = Normalize([EndPos[0] - RootPos[0], EndPos[1] - RootPos[1]])
    IntersectNodes = [RootNode, EndNode]
    CurrentNode = RootNode
    CurrentPos = RootPos
    while True:
        CurrentNode, CurrentPos = GetNextIntersectNode(CurrentNode, CurrentPos, Direction)
        if CurrentNode == EndNode:
            break
        IntersectNodes.append(CurrentNode)
    return IntersectNodes


class Building:
    def __init__(self, Index, Width, Length, Center):
        self.Index = Index
        self.Width = Width
        self.Length = Length
        self.Center = Center
        self.Nodes = []
        self.Borders = []

        self.__Direction = [0, 1]

    def RoadConquest(self):
        NearestRoadNode = GetNearestNode(self.Center, GetNodesInRadius(self.Center, self.Length, 1))
        if NearestRoadNode == [-1, -1]:
            return

        DistanceToRoad = 3
        self.__Direction = Normalize([self.Center[0] - NearestRoadNode[0], self.Center[1] - NearestRoadNode[1]])
        RootPos = [NearestRoadNode[0] + self.__Direction[0] * DistanceToRoad,
                   NearestRoadNode[1] + self.__Direction[1] * DistanceToRoad]

        self.RoadConquestOnOneSide(RootPos, [self.__Direction[1], -self.__Direction[0]], 4, self.Width / 8)
        self.RoadConquestOnOneSide(RootPos, [-self.__Direction[1], self.__Direction[0]], 4, self.Width / 8)

    def RoadConquestOnOneSide(self, RootPos, Direction, StepNum, StepLength):
        for i in range(StepNum):
            EndPos = [RootPos[0] + Direction[0] * StepLength, RootPos[1] + Direction[1] * StepLength]
            LoopTime = 0
            while True:
                IntersectNodes = GetIntersectNodes(RootPos, EndPos)
                TotalDistance = 0
                Flag = False
                for IntersectNode in IntersectNodes:
                    if Landmarks[IntersectNode[0], IntersectNode[1]] in [14, 13, 12, 11]:
                        TotalDistance += Landmarks[IntersectNode[0], IntersectNode[1]] - 10
                    elif Landmarks[IntersectNode[0], IntersectNode[1]] == 1:
                        Flag = True
                        break
                    else:
                        TotalDistance += 5
                if 2.5 < TotalDistance / len(IntersectNodes) and not Flag:
                    break
                EndPos = [EndPos[0] + self.__Direction[0], EndPos[1] + self.__Direction[1]]
                LoopTime += 1
                if LoopTime > 5:
                    return
            self.Borders.append([RootPos, EndPos])
            RootPos = EndPos

    def RoadConquestCheck(self):
        pass

    def RegionConquest(self):
        pass

    def RegionConquestCheck(self):
        pass

    def Draw(self, img, zoom):
        if len(self.Borders) == 0:
            return
        for Start, End in self.Borders:
            cv2.line(img, (int(Start[0] * zoom + zoom * 0.5), int(Start[1] * zoom + zoom * 0.5)),
                     (int(End[0] * zoom + zoom * 0.5), int(End[1] * zoom + zoom * 0.5)), (255, 255, 0), 2)


def mouse_event(event, x, y, flags, param):
    global zoom, mouse_state, img

    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_state == 1:
            img_copy = np.copy(img)
            building = Building(0, 30, 30, [x / zoom, y / zoom])
            building.RoadConquest()
            building.Draw(img_copy, zoom)
            cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('img', img_copy)
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_state = 1
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state = 0


if __name__ == '__main__':
    load_data_from_png('parcel_geo.png')
    img = cv2.imread('parcel_geo.png')
    DistanceTransformForRoad(img)
    img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    zoom = 5
    mouse_state = 0
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', mouse_event)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    # # 检查射线经过的像素的的情况
    # img = np.zeros((10, 10, 3), np.uint8)
    # for i in range(10):
    #     for j in range(10):
    #         img[i, j, :] = [0, 0, 0] if (i + j) % 2 == 0 else [255, 255, 255]
    #
    # StartPos = [3.2, 3]
    # EndPos = [4.8, 9]
    # Nodes = GetIntersectNodes(StartPos, EndPos)
    # for Node in Nodes:
    #     if Node[0] < 0 or Node[0] > 9 or Node[1] < 0 or Node[1] > 9:
    #         continue
    #     img[Node[1], Node[0], :] = [0, 0, 255]
    #
    # img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    # cv2.line(img, [round(StartPos[0] * 50 + 25), round(StartPos[1] * 50 + 25)],
    #          [round(EndPos[0] * 50 + 25), round(EndPos[1] * 50 + 25)], (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
