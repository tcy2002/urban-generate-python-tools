from fbxClass import *
from osm2png import *
from seed import *
from layout import *


def copy_test(old_name, copy_name):
    m = FBX(old_name)
    n = FBXCreator()
    old_mesh = m.get_class_nodes(fbx.FbxMesh.ClassId)[0]
    new_node = n.create_node('map')
    new_mesh = n.create_mesh('map_mesh', new_node)
    num_points = old_mesh.GetControlPointsCount()
    num_triangles = old_mesh.GetPolygonCount()
    print(num_points, num_triangles)

    new_mesh.SetControlPointCount(num_points)
    for i in range(num_points):
        point = old_mesh.GetControlPointAt(i)
        n.add_control_point(new_mesh, i, point)
    for i in range(num_triangles):
        triangle = [old_mesh.GetPolygonVertex(i, j) for j in range(3)]
        n.add_triangle(new_mesh, triangle)

    n.save(copy_name)


def create_line_rect(point1, point2, width=4):
    d = [point2[0] - point1[0], point2[1] - point1[1]]
    d = d / np.linalg.norm(d)
    n = [d[1], -d[0]]
    p1 = [point1[0] - n[0] * width / 2, point1[1] - n[1] * width / 2]
    p2 = [point1[0] + n[0] * width / 2, point1[1] + n[1] * width / 2]
    p3 = [point2[0] + n[0] * width / 2, point2[1] + n[1] * width / 2]
    p4 = [point2[0] - n[0] * width / 2, point2[1] - n[1] * width / 2]
    return [p1, p2, p3, p4]


def create_test(osm_name, fbx_name, platform='ue4'):
    # 上上周工作：osm地图转png
    roads = draw_osm(osm_name, r'.\tmp.png', 800)

    # 上周工作：生成png格式的建筑物布局
    seeds, _ = generate_seeds(r'.\tmp.png', 10, 20, 10)
    rects, _ = generate_layout(r'.\tmp.png', seeds)

    # 本周工作：生成fbx格式的建筑物布局并导入maya、ue4
    n = FBXCreator()
    new_node = n.create_node('map')
    new_mesh = n.create_mesh('map_mesh', new_node)

    # 计算顶点和三角形数目
    n_points = len(rects) * 4
    n_triangles = len(rects) * 2
    for road in roads:
        n_points += len(road) * 4
        n_triangles += len(road) * 2

    def add_mesh_rect(mesh, i, rect):
        for j in range(4):
            if platform == 'ue4':
                n.add_control_point(mesh, i + j, fbx.FbxVector4(rect[j][0], rect[j][1], 0))
                n.add_normal(mesh, i + j, fbx.FbxVector4(0, 0, 1))
            else:
                n.add_control_point(mesh, i + j, fbx.FbxVector4(rect[j][0], 0, rect[j][1]))
                n.add_normal(mesh, i + j, fbx.FbxVector4(0, 1, 0))
        n.add_polygon(mesh, [i - 4, i - 3, i - 2])
        n.add_polygon(mesh, [i - 2, i - 1, i - 4])

    new_mesh.SetControlPointCount(n_points)
    pi = 0
    for road in roads:
        for i in range(len(road) - 1):
            line_rect = create_line_rect(road[i], road[i + 1])
            add_mesh_rect(new_mesh, pi, line_rect)
            pi += 4
    for rect in rects:
        add_mesh_rect(new_mesh, pi, rect)
        pi += 4

    n.save(fbx_name)


if __name__ == '__main__':
    # copy_test(r'.\map.fbx', r'.\new_map.fbx')
    create_test(r'.\osmMaps\map3.osm', r'.\map3.fbx')
