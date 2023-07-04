import xml.etree.ElementTree as ET


class OsmParser:
    def __init__(self, path):
        self.path = path
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()

    def get_nodes(self):
        nodes = []
        for node in self.root.findall('node'):
            nodes.append(node)
        return nodes

    def get_ways(self):
        ways = []
        for way in self.root.findall('way'):
            ways.append(way)
        return ways

    def get_relations(self):
        relations = []
        for relation in self.root.findall('relation'):
            relations.append(relation)
        return relations

    def get_node_by_id(self, node_id):
        for node in self.root.findall('node'):
            if node.attrib['id'] == node_id:
                return node
        return None

    def get_way_by_id(self, way_id):
        for way in self.root.findall('way'):
            if way.attrib['id'] == way_id:
                return way
        return None

    def get_relation_by_id(self, relation_id):
        for relation in self.root.findall('relation'):
            if relation.attrib['id'] == relation_id:
                return relation
        return None

    def get_node_by_tag(self, tag_key, tag_value):
        for node in self.root.findall('node'):
            for tag in node.findall('tag'):
                if tag.attrib['k'] == tag_key and tag.attrib['v'] == tag_value:
                    return node
        return None

    def get_way_by_tag(self, tag_key, tag_value):
        for way in self.root.findall('way'):
            for tag in way.findall('tag'):
                if tag.attrib['k'] == tag_key and tag.attrib['v'] == tag_value:
                    return way
        return None

    def get_relation_by_tag(self, tag_key, tag_value):
        for relation in self.root.findall('relation'):
            for tag in relation.findall('tag'):
                if tag.attrib['k'] == tag_key and tag.attrib['v'] == tag_value:
                    return relation
        return None

    def get_nodes_by_tag_key(self, tag_key):
        nodes = []
        for node in self.root.findall('node'):
            for tag in node.findall('tag'):
                if tag.attrib['k'] == tag_key:
                    nodes.append(node)
        return nodes

    def get_ways_by_tag_key(self, tag_key):
        ways = []
        for way in self.root.findall('way'):
            for tag in way.findall('tag'):
                if tag.attrib['k'] == tag_key:
                    ways.append(way)
        return ways

    def get_relations_by_tag_key(self, tag_key):
        relations = []
        for relation in self.root.findall('relation'):
            for tag in relation.findall('tag'):
                if tag.attrib['k'] == tag_key:
                    relations.append(relation)
        return relations

    def get_nodes_by_tag_value(self, tag_value):
        nodes = []
        for node in self.root.findall('node'):
            for tag in node.findall('tag'):
                if tag.attrib['v'] == tag_value:
                    nodes.append(node)
        return nodes

    def get_ways_by_tag_value(self, tag_value):
        ways = []
        for way in self.root.findall('way'):
            for tag in way.findall('tag'):
                if tag.attrib['v'] == tag_value:
                    ways.append(way)
        return ways

    def get_relations_by_tag_value(self, tag_value):
        relations = []
        for relation in self.root.findall('relation'):
            for tag in relation.findall('tag'):
                if tag.attrib['v'] == tag_value:
                    relations.append(relation)
        return relations

    def get_nodes_by_tag_key_value(self, tag_key, tag_value):
        nodes = []
        for node in self.root.findall('node'):
            for tag in node.findall('tag'):
                if tag.attrib['k'] == tag_key and tag.attrib['v'] == tag_value:
                    nodes.append(node)
        return nodes

    def get_ways_by_tag_key_value(self, tag_key, tag_value):
        ways = []
        for way in self.root.findall('way'):
            for tag in way.findall('tag'):
                if tag.attrib['k'] == tag_key and tag.attrib['v'] == tag_value:
                    ways.append(way)
        return ways

    def get_relations_by_tag_key_value(self, tag_key, tag_value):
        relations = []
        for relation in self.root.findall('relation'):
            for tag in relation.findall('tag'):
                if tag.attrib['k'] == tag_key and tag.attrib['v'] == tag_value:
                    relations.append(relation)
        return relations

