
query = "(node(58.379593,26.7154817,58.3836303,26.7213126);<;);out;"

import overpy

api = overpy.Overpass()

# fetch all ways and nodes
result = api.query("""
    way(58.379593,26.7154817,58.3836303,26.7213126) ["highway"];
    (._;>;);
    out body;
    """)

for way in result.ways:
    print("Name: %s" % way.tags.get("name", "n/a"))
    print("  Highway: %s" % way.tags.get("highway", "n/a"))
    print("  Nodes:")
    for node in way.nodes:
        print("    id: %d, Lat: %f, Lon: %f" % (node.id, node.lat, node.lon))
        
