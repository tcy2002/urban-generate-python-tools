import numpy as np
import cv2
import threading
import time

# algorithm of parcel division based on geometry

# 0: Land, 1: Road, 3x: Building, 4:Occupied by building, 5: Virtual Border
Landmarks: np.ndarray
Buildings = []
lock = threading.Lock()


def load_data_from_png(path):
    global Landmarks, Buildings
    img = cv2.imread(path)
    Landmarks = np.zeros((img.shape[1], img.shape[0]), np.uint8)
    roads = np.where((img == [255, 255, 255]).all(axis=2))
    buildings1 = np.where((img == [0, 255, 0]).all(axis=2))
    buildings2 = np.where((img == [255, 0, 0]).all(axis=2))
    buildings3 = np.where((img == [0, 0, 255]).all(axis=2))
    buildings4 = np.where((img == [0, 255, 255]).all(axis=2))
    Landmarks[roads[1], roads[0]] = 1
    for i in range(len(buildings1[0])):
        Buildings.append(FBuilding(i, 15, 15, [buildings1[1][i], buildings1[0][i]]))
    size = len(Buildings)
    for i in range(len(buildings2[0])):
        Buildings.append(FBuilding(i + size, 25, 25, [buildings2[1][i], buildings2[0][i]]))
    size = len(Buildings)
    for i in range(len(buildings3[0])):
        Buildings.append(FBuilding(i + size, 50, 50, [buildings3[1][i], buildings3[0][i]]))
    size = len(Buildings)
    for i in range(len(buildings4[0])):
        Buildings.append(FBuilding(i + size, 120, 120, [buildings4[1][i], buildings4[0][i]]))


def MarkNeighborsOfAllBuildings():
    global Buildings
    for Building in Buildings:
        for Other in Buildings:
            if Building.Index != Other.Index and \
                    Distance(Building.Center, Other.Center) < 2 * Building.Length:
                IntersectNodes = GetIntersectNodes(Building.Center, Other.Center)
                Flag = False
                for Node in IntersectNodes:
                    if CheckNode(Node) and Landmarks[Node[0], Node[1]] == 1:
                        Flag = True
                        break
                if Flag:
                    continue
                Building.AddNeighbor(Other.Index)
                Other.AddNeighbor(Building.Index)


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


def CheckNode(Node):
    global Landmarks
    x, y = Landmarks.shape
    return 0 <= Node[0] < x and 0 <= Node[1] < y


def Distance(Node1, Node2):
    return np.sqrt((Node1[0] - Node2[0]) * (Node1[0] - Node2[0]) +
                   (Node1[1] - Node2[1]) * (Node1[1] - Node2[1]))


def Dot(Node1, Node2):
    return Node1[0] * Node2[0] + Node1[1] * Node2[1]


def Lerp(Node1, Node2, Rate):
    return [Node1[0] * Rate + Node2[0] * (1 - Rate), Node1[1] * Rate + Node2[1] * (1 - Rate)]


def Average(Nodes):
    if len(Nodes) == 0:
        return [0, 0]
    return [sum([x for x, _ in Nodes]) / len(Nodes), sum([y for _, y in Nodes]) / len(Nodes)]


def Magnitude(Node):
    return np.sqrt(Node[0] * Node[0] + Node[1] * Node[1])


def Normalize(Node):
    m = Magnitude(Node)
    return [Node[0] / m, Node[1] / m]


def Projection(Node, Root, Normal):
    Direction = [Root[0] - Node[0], Root[1] - Node[1]]
    Length = Direction[0] * Normal[0] + Direction[1] * Normal[1]
    return [Node[0] + Normal[0] * Length, Node[1] + Normal[1] * Length]


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


def GetNodesInRadius(Node, Radius, Types):
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
            if Landmarks[Col, Row] in Types:
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


def GetDistanceOnDirection(RootPos, Direction, MaxDistance, Types):
    CurrentNode = [round(RootPos[0]), round(RootPos[1])]
    CurrentPos = RootPos
    while True:
        CurrentNode, CurrentPos = GetNextIntersectNode(CurrentNode, CurrentPos, Direction)
        Length = Distance(CurrentPos, RootPos)
        if Length > MaxDistance:
            return MaxDistance
        if not CheckNode(CurrentNode) or Landmarks[CurrentNode[0], CurrentNode[1]] in Types:
            return Length


def GetIntersectNodes(RootPos, EndPos):
    # if EndPos[0] % 0.5 == 0.0:
    #     EndPos[0] += 0.001
    # if EndPos[1] % 0.5 == 0.0:
    #     EndPos[1] += 0.001
    EndPos[0] += 0.001
    EndPos[1] += 0.001

    RootNode = [round(RootPos[0]), round(RootPos[1])]
    EndNode = [round(EndPos[0]), round(EndPos[1])]
    Direction = Normalize([EndPos[0] - RootPos[0], EndPos[1] - RootPos[1]])
    IntersectNodes = [RootNode]
    CurrentNode = RootNode
    CurrentPos = RootPos
    while True:
        if CurrentNode == EndNode:
            break
        CurrentNode, CurrentPos = GetNextIntersectNode(CurrentNode, CurrentPos, Direction)
        IntersectNodes.append(CurrentNode)
    return IntersectNodes


class FBuilding:
    def __init__(self, Index, Width, Length, Center):
        self.Index = Index
        self.Width = Width
        self.Length = Length
        self.Center = Center
        self.Nodes = []
        self.Borders = []
        self.Redo = False

        self.__Valid = True
        self.__Direction = [0, 1]
        self.__RoadEndPos1 = [0, 0]
        self.__RoadEndPos2 = [0, 0]
        self.__Neighbors = dict()
        self.__VirtualBorderNodes = []
        self.__NearestRoadNode = GetNearestNode(Center, GetNodesInRadius(Center, Length, [1]))

    def AddNeighbor(self, Index):
        global Buildings
        self.__Neighbors[Index] = Buildings[Index]

    def CheckLine(self, StartPos, EndPos):
        IntersectNodes = GetIntersectNodes(StartPos, EndPos)
        for IntersectNode in IntersectNodes:
            if not CheckNode(IntersectNode) or Landmarks[IntersectNode[0], IntersectNode[1]] in [1, 5]:
                return False
        EndPos = [round(EndPos[0]), round(EndPos[1])]
        NearByNodes = Get4Neighbors(EndPos)
        for NearByNode in NearByNodes:
            if Landmarks[NearByNode[0], NearByNode[1]] in [1, 5]:
                return False
        return True

    def DrawLine(self, img, zoom, color, width):
        Size = len(self.Borders)
        if Size == 0:
            return
        for i in range(0, Size, 2):
            cv2.line(img, (int(self.Borders[i][0] * zoom + zoom * 0.5), int(self.Borders[i][1] * zoom + zoom * 0.5)),
                     (int(self.Borders[i + 1][0] * zoom + zoom * 0.5), int(self.Borders[i + 1][1] * zoom + zoom * 0.5)),
                     color, width)

    def Resample(self, Segments, StepLength):
        TotalLength = 0
        Size = len(Segments)
        if Size == 0:
            return []
        Lengths = [0.0] * (Size // 2)
        for i in range(0, Size, 2):
            Length = Distance(Segments[i], Segments[i + 1])
            TotalLength += Length
            Lengths[i // 2] = Length

        StepLength = TotalLength / max(round(TotalLength / StepLength), 1)
        CurrentLength = 0
        CurrentIndex = 0
        NextStart = StartPos = Segments[0]
        ResampledSegments = []
        while True:
            Length = Lengths[CurrentIndex // 2]
            if CurrentLength + Length < StepLength - 0.001:
                CurrentLength += Length
                CurrentIndex += 2
                if CurrentIndex >= Size:
                    break
                NextStart = Segments[CurrentIndex]
            else:
                Rate = (StepLength - CurrentLength) / Length
                EndPos = Lerp(NextStart, Segments[CurrentIndex + 1], 1 - Rate)
                ResampledSegments.append(StartPos)
                ResampledSegments.append(EndPos)
                Lengths[CurrentIndex // 2] = Length * (1 - Rate)
                CurrentLength = 0
                NextStart = StartPos = EndPos

        return ResampledSegments

    def Generate(self):
        self.Init()
        LeftRoad, RightRoad = self.RoadConquest()
        LeftSide, RightSide, TopSide = self.RegionConquest()
        self.End()

        # self.Borders.extend(LeftRoad)
        # self.Borders.extend(RightRoad)
        # self.Borders.extend(LeftSide)
        # self.Borders.extend(RightSide)
        # self.Borders.extend(TopSide)
        self.Borders.extend(self.Resample(LeftRoad, 5))
        self.Borders.extend(self.Resample(RightRoad, 5))
        self.Borders.extend(self.Resample(LeftSide, 5))
        self.Borders.extend(self.Resample(RightSide, 5))
        self.Borders.extend(self.Resample(TopSide, 5))

    def Init(self):
        for Neighbor in self.__Neighbors.values():
            if not Neighbor.__Valid:
                continue

            Rate = self.Length / (self.Length + Neighbor.Length)
            Center = Lerp(self.Center, Neighbor.Center, 1 - Rate)
            Direction = Normalize([self.Center[1] - Neighbor.Center[1], Neighbor.Center[0] - self.Center[0]])
            Length = max(self.Length, Neighbor.Length)
            Start = [Center[0] - Direction[0] * Length, Center[1] - Direction[1] * Length]

            for i in range(Length * 2 + 1):
                Node = [round(Start[0] + Direction[0] * i), round(Start[1] + Direction[1] * i)]
                if CheckNode(Node) and Landmarks[Node[0], Node[1]] == 0:
                    self.__VirtualBorderNodes.append(Node)
                    Landmarks[Node[0], Node[1]] = 5

    def End(self):
        global Landmarks
        for Node in self.__VirtualBorderNodes:
            Landmarks[Node[0], Node[1]] = 0
        self.__VirtualBorderNodes.clear()

    def RoadConquest(self):
        global del_num
        if not self.__Valid:
            return [], []
        if self.__NearestRoadNode != [-1, -1]:
            self.__Direction = Normalize([self.Center[0] - self.__NearestRoadNode[0],
                                          self.Center[1] - self.__NearestRoadNode[1]])
            RootPos = [self.__NearestRoadNode[0] + self.__Direction[0] * 2,
                       self.__NearestRoadNode[1] + self.__Direction[1] * 2]
        else:
            self.__Direction = [0, 1]
            RootPos = [self.Center[0],
                       (self.Center[1] - self.Length / 2) if self.Center[1] - self.Length / 2 > 0 else 0]

        StepNum = 4
        StepNum1, self.__RoadEndPos1, Borders1 = self.RoadConquestOnOneSide(RootPos,
                                                                            [self.__Direction[1], -self.__Direction[0]],
                                                                            StepNum, self.Width / (StepNum * 2))
        StepNum2, self.__RoadEndPos2, Borders2 = self.RoadConquestOnOneSide(RootPos,
                                                                            [-self.__Direction[1], self.__Direction[0]],
                                                                            StepNum, self.Width / (StepNum * 2))

        if StepNum1 < StepNum and StepNum2 == StepNum:
            _, self.__RoadEndPos2, Borders2S = self.RoadConquestOnOneSide(self.__RoadEndPos2,
                                                                          [-self.__Direction[1], self.__Direction[0]],
                                                                          (StepNum - StepNum1) * 2,
                                                                          self.Width / (StepNum * 4))
            Borders2.extend(Borders2S)
        elif StepNum1 == StepNum and StepNum2 < StepNum:
            _, self.__RoadEndPos1, Borders1S = self.RoadConquestOnOneSide(self.__RoadEndPos1,
                                                                          [self.__Direction[1], -self.__Direction[0]],
                                                                          (StepNum - StepNum2) * 2,
                                                                          self.Width / (StepNum * 4))
            Borders1.extend(Borders1S)
        elif StepNum1 + StepNum2 < StepNum:
            self.__Valid = False
            del_num += 1
            for Neighbor in self.__Neighbors.values():
                Neighbor.Redo = True
            return [], []
        return Borders1, Borders2

    def RoadConquestOnOneSide(self, RootPos, Direction, StepNum, StepLength):
        Borders = []
        for i in range(StepNum):
            EndPos = [RootPos[0] + Direction[0] * StepLength, RootPos[1] + Direction[1] * StepLength]
            LoopTime = 0
            while True:
                if self.CheckLine(RootPos, EndPos):
                    break
                LoopTime += 1
                if LoopTime > StepLength:
                    return i, RootPos, Borders
                EndPos = [EndPos[0] + self.__Direction[0], EndPos[1] + self.__Direction[1]]
            Borders.append(RootPos)
            Borders.append(EndPos)
            RootPos = EndPos
        return StepNum, RootPos, Borders

    def RegionConquest(self):
        if not self.__Valid:
            return [], [], []

        Diff = [self.__RoadEndPos1[0] - self.__RoadEndPos2[0], self.__RoadEndPos1[1] - self.__RoadEndPos2[1]]
        DiffOnDirection = Diff[0] * self.__Direction[0] + Diff[1] * self.__Direction[1]
        TopPos1, Borders1 = self.RegionConquestOnOneSide(self.__RoadEndPos1,
                                                         [-self.__Direction[1], self.__Direction[0]],
                                                         self.Length - DiffOnDirection / 2)
        TopPos2, Borders2 = self.RegionConquestOnOneSide(self.__RoadEndPos2,
                                                         [self.__Direction[1], -self.__Direction[0]],
                                                         self.Length + DiffOnDirection / 2)

        Length = Distance(TopPos1, TopPos2)
        if Length < 1:
            return Borders1, Borders2, []
        MiddlePos = [(self.__RoadEndPos1[0] + self.__RoadEndPos2[0]) / 2,
                     (self.__RoadEndPos1[1] + self.__RoadEndPos2[1]) / 2]
        MiddlePos = [MiddlePos[0] + self.__Direction[0] * self.Length,
                     MiddlePos[1] + self.__Direction[1] * self.Length]
        Borders3 = self.RegionConquestOnTop(TopPos1, TopPos2, MiddlePos, round(Length / (self.Width / 5)) + 1)
        return Borders1, Borders2, Borders3

    def RegionConquestOnOneSide(self, RootPos, Normal, Length):
        TopPos, Found, MiddlePos = self.RegionConquestFindTopPos(RootPos, Normal, 6, Length / 6)
        Borders = []
        if not Found:
            MaxAngle = 40
            MaxDistance = Length * np.sin(np.radians(MaxAngle))
            RoadNodes = GetNodesInRadius(TopPos, MaxDistance, [1, 5])
            if len(RoadNodes) == 0:
                Borders.append(RootPos)
                Borders.append(TopPos)
                return TopPos, Borders
            Found, MiddlePos = self.RegionConquestFindMiddlePos(RootPos, TopPos, Normal, MaxAngle)
        if Found:
            Borders.append(TopPos)
            Borders.append(MiddlePos)

        EndPos = MiddlePos if Found else TopPos
        Found, InterPos = self.RegionConquestFindInterPos(RootPos, EndPos, Normal, self.Width / 2, 6)

        if Found:
            Borders.append(EndPos)
            Borders.append(InterPos)
            Borders.append(InterPos)
            Borders.append(RootPos)
        else:
            Borders.append(EndPos)
            Borders.append(RootPos)
        return TopPos, Borders

    def RegionConquestOnTop(self, Pos1, Pos2, MiddlePos, StepNum):
        Direction = Normalize([Pos2[0] - Pos1[0], Pos2[1] - Pos1[1]])
        Sin = abs(Direction[0] * self.__Direction[1] - Direction[1] * self.__Direction[0])
        Length = Distance(Pos1, Pos2) * Sin
        StepLength = Length / StepNum

        StartPos = Projection(Pos1, MiddlePos, self.__Direction)
        StartDirection = [-self.__Direction[1], self.__Direction[0]]
        if StartDirection[0] * Direction[0] + StartDirection[1] * Direction[1] < 0:
            StartDirection = [-StartDirection[0], -StartDirection[1]]

        LastPos = Pos1
        Borders = []
        for i in range(1, StepNum + 1):
            Pos = [StartPos[0] + StartDirection[0] * StepLength * i,
                   StartPos[1] + StartDirection[1] * StepLength * i] if i < StepNum else Pos2
            LoopTime = 0
            while True:
                LoopTime += 1
                if self.CheckLine(LastPos, Pos) or LoopTime > self.Length:
                    break
                Pos = [Pos[0] - self.__Direction[0], Pos[1] - self.__Direction[1]]
            Borders.append(LastPos)
            Borders.append(Pos)
            LastPos = Pos
        return Borders

    def RegionConquestFindTopPos(self, RootPos, Direction, StepNum, StepLength):
        StartPos = RootPos
        Points = []
        Flag = True
        EndFlag = False

        for i in range(StepNum):
            EndPos = [RootPos[0] + self.__Direction[0] * StepLength * (i + 1),
                      RootPos[1] + self.__Direction[1] * StepLength * (i + 1)]
            LoopTime = 0
            while True:
                if self.CheckLine(StartPos, EndPos):
                    break
                else:
                    Flag = False
                LoopTime += 1
                if LoopTime > self.Width / 2:
                    EndFlag = True
                    break
                EndPos = [EndPos[0] + Direction[0], EndPos[1] + Direction[1]]
            if EndFlag:
                break
            Points.append([EndPos, LoopTime])
            StartPos = EndPos

        if Flag:
            return StartPos, False, [-1, -1]
        MaxValue = 0
        MaxIndex = -1
        for i in range(len(Points)):
            if Points[i][1] >= MaxValue:
                MaxValue = Points[i][1]
                MaxIndex = i
        if MaxIndex == len(Points) - 1:
            return StartPos, False, [-1, -1]
        return StartPos, True, Points[MaxIndex][0]

    def RegionConquestFindMiddlePos(self, RootPos, TopPos, Normal, MaxAngle):
        FirstFlag = True

        while MaxAngle > 0:
            Rad = np.radians(MaxAngle)
            Tan = np.tan(Rad)
            Cos = np.cos(Rad)
            Len = Distance(TopPos, RootPos)
            R = Normalize([TopPos[0] - RootPos[0] + Len * Tan * -Normal[0],
                           TopPos[1] - RootPos[1] + Len * Tan * -Normal[1]])
            R = [R[0] * Len * Cos, R[1] * Len * Cos]
            MiddlePos = [RootPos[0] + R[0], RootPos[1] + R[1]]
            if self.CheckLine(RootPos, MiddlePos):
                if FirstFlag:
                    return False, TopPos
                return True, MiddlePos
            FirstFlag = False
            MaxAngle -= 5

        return False, TopPos

    def RegionConquestFindInterPos(self, RootPos, EndPos, Normal, MaxDistance, StepNum):
        BorderLength = Distance(RootPos, EndPos)
        BorderDirection = Normalize([EndPos[0] - RootPos[0], EndPos[1] - RootPos[1]])
        BorderNormal = [-BorderDirection[1], BorderDirection[0]]
        if BorderNormal[0] * Normal[0] + BorderNormal[1] * Normal[1] > 0:
            BorderNormal = [-BorderNormal[0], -BorderNormal[1]]
        InterPoints = [[RootPos[0] + BorderDirection[0] * i * BorderLength / StepNum,
                        RootPos[1] + BorderDirection[1] * i * BorderLength / StepNum] for i in range(1, StepNum)]
        InterDistances = [GetDistanceOnDirection(Pos, BorderNormal, MaxDistance, [1, 5]) for Pos in InterPoints]

        if min(InterDistances) == MaxDistance:
            return False, EndPos

        MaxIndex = InterDistances.index(max(InterDistances))
        InterPoint = InterPoints[MaxIndex]
        LoopTime = 0
        while True:
            TmpPoint = [InterPoint[0] + BorderNormal[0], InterPoint[1] + BorderNormal[1]]
            if not self.CheckLine(RootPos, TmpPoint) or not self.CheckLine(EndPos, TmpPoint):
                return True, InterPoint
            LoopTime += 1
            if LoopTime > MaxDistance:
                return True, TmpPoint
            InterPoint = TmpPoint


def mouse_event(event, x, y, flags, param):
    global zoom, mouse_state, img

    if lock.locked():
        return

    lock.acquire()
    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_state == 1:
            print(x, y)
            img_copy = np.copy(img)
            building = FBuilding(0, 30, 30, [x / zoom, y / zoom])
            building.RoadConquest()
            building.RegionConquest()
            building.DrawLine(img_copy, zoom, (255, 255, 0), 2)
            cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('img', img_copy)
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_state = 1
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state = 0
    lock.release()


def load_raw_png(path, out_path):
    img = cv2.imread(path)
    img[img != 255] = 0
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imwrite(out_path, img)


if __name__ == '__main__':
    # load_raw_png('parcel_real_raw.png', 'parcel_real_h2.png')

    total_time = 0.0
    start = time.time()
    load_data_from_png('parcel_real2.png')
    img = cv2.imread('parcel_real2.png')
    MarkNeighborsOfAllBuildings()
    end = time.time()
    print('building num: ', len(Buildings))
    print('load data time: ', end - start, 's')
    total_time += end - start

    start = time.time()
    del_num = 0
    # indices = [118]
    indices = range(0, len(Buildings))
    for index in indices:
        Buildings[index].Generate()
    end = time.time()
    print('generate time: ', end - start, 's')
    total_time += end - start

    start = time.time()
    Num = 0
    for index in indices:
        if Buildings[index].Redo:
            Num += 1
            Buildings[index].Borders.clear()
            Buildings[index].Generate()
            Buildings[index].Redo = False
    end = time.time()
    print('rejected num: ', del_num)
    print('regenerate num: ', Num)
    print('regenerate time: ', end - start, 's')
    total_time += end - start

    start = time.time()
    for index in indices:
        if Buildings[index].Width == 15:
            color = (255, 255, 0)
        elif Buildings[index].Width == 25:
            color = (255, 200, 150)
        elif Buildings[index].Width == 50:
            color = (200, 200, 255)
        else:
            color = (0, 200, 255)
        Buildings[index].DrawLine(img, 1, color, 1)
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    for index in indices:
        if Buildings[index].Width == 15:
            color = (0, 255, 0)
        elif Buildings[index].Width == 25:
            color = (255, 0, 0)
        elif Buildings[index].Width == 50:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        cv2.circle(img, (int(Buildings[index].Center[0] * 2), int(Buildings[index].Center[1] * 2)), 3, color, -1)
    end = time.time()
    print('draw time: ', end - start, 's')
    total_time += end - start
    print('total time: ', total_time, 's')

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imwrite('geo_divide.png', img)

    # # 单个地块生成测试
    # load_data_from_png('parcel_geo.png')
    # img = cv2.imread('parcel_geo.png')
    # img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    # zoom = 5
    # mouse_state = 0
    # cv2.namedWindow('img')
    # cv2.setMouseCallback('img', mouse_event)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # # pos = (440, 377)
    # # building = FBuilding(0, 30, 30, (pos[0] / zoom, pos[1] / zoom))
    # # building.RoadConquest()
    # # building.RegionConquest()
    # # building.DrawLine(img, zoom, (255, 255, 0), 2)
    # # cv2.circle(img, pos, 3, (0, 0, 255), -1)
    # # cv2.imshow('img', img)
    # # cv2.waitKey(0)

    # # 检查射线经过的像素的的情况
    # img = np.zeros((10, 10, 3), np.uint8)
    # for i in range(10):
    #     for j in range(10):
    #         img[i, j, :] = [0, 0, 0] if (i + j) % 2 == 0 else [255, 255, 255]
    #
    # StartPos = [440.3305140209734, 324.9424632221321]
    # EndPos = [439.6986952374362, 325.3503742964159]
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
