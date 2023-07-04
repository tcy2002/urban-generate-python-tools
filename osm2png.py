import numpy as np
from osmParser import OsmParser
import math
import cv2


# convert lat/lon to x/y
def _latlon_to_xy(lat, lon):
    lat = lat * math.pi / 180
    lon = lon * math.pi / 180
    x = 6378137.0 * lon
    y = -6378137.0 * math.log(math.tan((math.pi / 4) + (lat / 2)))
    return x, y


# get all roads
def _get_roads(osm_file):
    file = OsmParser(osm_file)
    ways = file.get_ways_by_tag_key('highway')

    roads = []
    for way in ways:
        roads.append([])
        nds = way.findall('nd')
        for nd in nds:
            nodeId = nd.attrib['ref']
            node = file.get_node_by_id(nodeId)
            if node is not None:
                lat = float(node.attrib['lat'])
                lon = float(node.attrib['lon'])
                roads[-1].append(_latlon_to_xy(lat, lon))

    return roads


# normalize coordinates
def _normalize(coords, size):
    xStart, yStart = 1e8, 1e8
    xEnd, yEnd = -1e8, -1e8

    for r in coords:
        for x, y in r:
            xStart = min(xStart, x)
            yStart = min(yStart, y)
            xEnd = max(xEnd, x)
            yEnd = max(yEnd, y)
    xLen = size
    yLen = xLen * (yEnd - yStart) / (xEnd - xStart)

    normalized = []
    for r in coords:
        r = [(x - xStart, y - yStart) for x, y in r]
        r = [(x / (xEnd - xStart) * xLen, y / (yEnd - yStart) * yLen) for x, y in r]
        normalized.append(r)

    return normalized, (int(xLen), int(yLen))


# draw roads on image
def _draw_roads(img, roads):
    for road in roads:
        for i in range(len(road) - 1):
            x1, y1 = road[i]
            x2, y2 = road[i + 1]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)

    return img


# main
def draw_osm(osm_file, img_file, size):
    roads = _get_roads(osm_file)
    roads, shape = _normalize(roads, size)
    img = np.zeros((shape[1], shape[0], 3), np.uint8)
    img = _draw_roads(img, roads)
    cv2.imwrite(img_file, img)
    return roads
