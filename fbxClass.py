import sys
sys.path.append(r'FBX\Python37_x64')

import FbxCommon
import fbx


class FBX(object):
    """
    A Helper class to load and maintain fbx scenes.
    Modified from https://gist.github.com/Meatplowz/8f408912cf554f2d11085fb68b62d3a3
    """

    def __init__(self, filename):
        """
        FBX Scene Object
        """
        self.filename = filename
        self.sdk_manager, self.scene = FbxCommon.InitializeSdkObjects()
        FbxCommon.LoadScene(self.sdk_manager, self.scene, filename)

        self.root_node = self.scene.GetRootNode()
        self.scene_nodes = self.get_scene_nodes()

    def close(self):
        """
        You need to run this to close the FBX scene safely
        """
        # destroy objects created by the sdk
        self.sdk_manager.Destroy()

    def __get_scene_nodes_recursive(self, node):
        """
        Rescursive method to get all scene nodes
        this should be private, called by get_scene_nodes()
        """
        self.scene_nodes.append(node)
        for i in range(node.GetChildCount()):
            self.__get_scene_nodes_recursive(node.GetChild(i))

    def __cast_property_type(self, fbx_property):
        """
        Cast a property to type to properly get the value
        """
        casted_property = None

        unsupported_types = [fbx.eFbxUndefined, fbx.eFbxChar, fbx.eFbxUChar, fbx.eFbxShort, fbx.eFbxUShort,
                             fbx.eFbxUInt,
                             fbx.eFbxLongLong, fbx.eFbxHalfFloat, fbx.eFbxDouble4x4, fbx.eFbxEnum, fbx.eFbxTime,
                             fbx.eFbxReference, fbx.eFbxBlob, fbx.eFbxDistance, fbx.eFbxDateTime, fbx.eFbxTypeCount]

        # property is not supported or mapped yet
        property_type = fbx_property.GetPropertyDataType().GetType()
        if property_type in unsupported_types:
            return None

        if property_type == fbx.eFbxBool:
            casted_property = fbx.FbxPropertyBool1(fbx_property)
        elif property_type == fbx.eFbxDouble:
            casted_property = fbx.FbxPropertyDouble1(fbx_property)
        elif property_type == fbx.eFbxDouble2:
            casted_property = fbx.FbxPropertyDouble2(fbx_property)
        elif property_type == fbx.eFbxDouble3:
            casted_property = fbx.FbxPropertyDouble3(fbx_property)
        elif property_type == fbx.eFbxDouble4:
            casted_property = fbx.FbxPropertyDouble4(fbx_property)
        elif property_type == fbx.eFbxInt:
            casted_property = fbx.FbxPropertyInteger1(fbx_property)
        elif property_type == fbx.eFbxFloat:
            casted_property = fbx.FbxPropertyFloat1(fbx_property)
        elif property_type == fbx.eFbxString:
            casted_property = fbx.FbxPropertyString(fbx_property)
        else:
            print('Unknown property type in `__cast_property_type`')
            exit(-1)

        return casted_property

    def get_scene_nodes(self):
        """
        Get all nodes in the fbx scene
        """
        self.scene_nodes = []
        for i in range(self.root_node.GetChildCount()):
            self.__get_scene_nodes_recursive(self.root_node.GetChild(i))
        return self.scene_nodes

    def get_type_nodes(self, type, node=None):
        """
        Get nodes from the scene with the given type
        display_layer_nodes = fbx_file.get_type_nodes( u'DisplayLayer' )
        """
        nodes = []
        if node is None:
            node = self.scene

        num_objects = node.GetSrcObjectCount()
        for i in range(0, num_objects):
            n = node.GetSrcObject(i)
            if n:
                if n.GetTypeName() == type:
                    nodes.append(n)
        return nodes

    def get_class_nodes(self, class_id, node=None):
        """
        Get nodes in the scene with the given classid
        geometry_nodes = fbx_file.get_class_nodes( fbx.FbxGeometry.ClassId )
        """
        nodes = []
        if node is None:
            node = self.scene

        num_nodes = node.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(class_id))
        for index in range(0, num_nodes):
            n = node.GetSrcObject(fbx.FbxCriteria.ObjectType(class_id), index)
            if n:
                nodes.append(n)
        return nodes

    def get_property(self, node, property_string):
        """
        Gets a property from an Fbx node
        export_property = fbx_file.get_property(node, 'no_export')
        """
        fbx_property = node.FindProperty(property_string)
        return fbx_property

    def get_property_value(self, node, property_string):
        """
        Gets the property value from an Fbx node
        property_value = fbx_file.get_property_value(node, 'no_export')
        """
        fbx_property = node.FindProperty(property_string)
        if fbx_property.IsValid():
            # cast to correct property type so you can get
            casted_property = self.__cast_property_type(fbx_property)
            if casted_property:
                return casted_property.Get()
        return None

    def get_node_by_name(self, name):
        """
        Get the fbx node by name
        """
        self.get_scene_nodes()
        # right now this is only getting the first one found
        node = [node for node in self.scene_nodes if node.GetName() == name]
        if node:
            return node[0]
        return None

    def remove_namespace(self):
        """
        Remove all namespaces from all nodes
        This is not an ideal method but
        """
        self.get_scene_nodes()
        for node in self.scene_nodes:
            orig_name = node.GetName()
            split_by_colon = orig_name.split(':')
            if len(split_by_colon) > 1:
                new_name = split_by_colon[-1:][0]
                node.SetName(new_name)
        return True

    def remove_node_property(self, node, property_string):
        """
        Remove a property from an Fbx node
        remove_property = fbx_file.remove_property(node, 'UDP3DSMAX')
        """
        node_property = self.get_property(node, property_string)
        if node_property.IsValid():
            node_property.DestroyRecursively()
            return True
        return False

    def remove_nodes_by_names(self, names):
        """
        Remove nodes from the fbx file from a list of names
        names = ['object1','shape2','joint3']
        remove_nodes = fbx_file.remove_nodes_by_names(names)
        """

        if names is None or len(names) == 0:
            return True

        self.get_scene_nodes()
        remove_nodes = [node for node in self.scene_nodes if node.GetName() in names]
        for node in remove_nodes:
            disconnect_node = self.scene.DisconnectSrcObject(node)
            remove_node = self.scene.RemoveNode(node)
        self.get_scene_nodes()
        return True

    def save(self, filename=None):
        """
        Save the current fbx scene as the incoming filename .fbx
        """
        # save as a different filename
        if filename is not None:
            FbxCommon.SaveScene(self.sdk_manager, self.scene, filename)
        else:
            FbxCommon.SaveScene(self.sdk_manager, self.scene, self.filename)
        self.close()


class FBXCreator:
    """
    Class to create FBX files
    """
    def __init__(self):
        self.sdk_manager, self.scene = FbxCommon.InitializeSdkObjects()
        self.root_node = self.scene.GetRootNode()

    def close(self):
        """
        You need to run this to close the FBX scene safely
        """
        # destroy objects created by the sdk
        self.sdk_manager.Destroy()

    def create_node(self, name):
        """
        Create a node on the root node
        """
        node = fbx.FbxNode.Create(self.scene, name)
        self.root_node.AddChild(node)
        return node

    def create_mesh(self, name, node):
        """
        Create a mesh on the root node
        """
        mesh = fbx.FbxMesh.Create(self.scene, name)
        node.AddNodeAttribute(mesh)
        return mesh

    def add_control_point(self, mesh, i, point):
        """
        Add a control point to the mesh
        """
        mesh.SetControlPointAt(point, i)

    def add_polygon(self, mesh, vertices):
        """
        Add a triangle to the mesh
        """
        mesh.BeginPolygon()
        for v in vertices:
            mesh.AddPolygon(v)
        mesh.EndPolygon()

    def add_normal(self, mesh, i, normal):
        """
        Add a normal to the mesh
        """
        mesh.SetControlPointNormalAt(normal, i)

    def save(self, filename):
        """
        Save the current fbx scene
        """
        FbxCommon.SaveScene(self.sdk_manager, self.scene, filename)
        self.close()
