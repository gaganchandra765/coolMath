"""
trav_animation.py

Creates an MP4 that animates BFS then DFS traversal over a simple city graph on a map of India.
- Outputs: traversal.mp4

Requires: geopandas, matplotlib, networkx, shapely, pyproj
ffmpeg must be installed on your system (you said you have it).
"""

import os
import urllib.request
import math
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point
import matplotlib.animation as animation

# ---------------------------
# 1) Download Natural Earth geojson (if missing)
# ---------------------------
GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)
MAP_FILE = "countries.geojson"
if not os.path.exists(MAP_FILE):
    print("Downloading country boundaries (one-time)...")
    urllib.request.urlretrieve(GEOJSON_URL, MAP_FILE)
    print("Downloaded:", MAP_FILE)

# ---------------------------
# 2) Define 20 major Indian cities (lon, lat)
# ---------------------------
cities = {
    "Delhi": (77.1025, 28.7041),
    "Mumbai": (72.8777, 19.0760),
    "Kolkata": (88.3639, 22.5726),
    "Chennai": (80.2707, 13.0827),
    "Bengaluru": (77.5946, 12.9716),
    "Hyderabad": (78.4867, 17.3850),
    "Jaipur": (75.7873, 26.9124),
    "Ahmedabad": (72.5714, 23.0225),
    "Pune": (73.8567, 18.5204),
    "Lucknow": (80.9462, 26.8467),
    "Surat": (72.8311, 21.1702),
    "Chandigarh": (76.7794, 30.7333),
    "Bhopal": (77.4126, 23.2599),
    "Indore": (75.8577, 22.7196),
    "Patna": (85.1376, 25.5941),
    "Guwahati": (91.7450, 26.1445),
    "Coimbatore": (76.9573, 11.0168),
    "Kochi": (76.2673, 9.9312),
    "Nagpur": (79.0882, 21.1458),
    "Visakhapatnam": (83.2185, 17.6868)
}

# ---------------------------
# 3) Create graph by connecting each city to its k nearest neighbors
# ---------------------------
def haversine(lon1, lat1, lon2, lat2):
    # returns distance in kilometers
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# build networkx graph
G = nx.Graph()
for name, (lon, lat) in cities.items():
    G.add_node(name, lon=lon, lat=lat)

# connect to k nearest neighbors
k = 3
city_items = list(cities.items())
for i, (name_i, (lon_i, lat_i)) in enumerate(city_items):
    # compute distances to others
    dists = []
    for j, (name_j, (lon_j, lat_j)) in enumerate(city_items):
        if i == j:
            continue
        d = haversine(lon_i, lat_i, lon_j, lat_j)
        dists.append((d, name_j))
    dists.sort()
    for _, neighbor in dists[:k]:
        if not G.has_edge(name_i, neighbor):
            G.add_edge(name_i, neighbor, weight=haversine(
                lon_i, lat_i, cities[neighbor][0], cities[neighbor][1]
            ))

# ---------------------------
# 4) BFS and DFS traversal orders
# ---------------------------
def bfs_order(graph, start):
    visited = set([start])
    queue = [start]
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for n in sorted(graph.neighbors(v)):
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return order

def dfs_order(graph, start):
    visited = set()
    order = []
    def dfs(v):
        visited.add(v)
        order.append(v)
        for n in sorted(graph.neighbors(v)):
            if n not in visited:
                dfs(n)
    dfs(start)
    return order

START = "Delhi"
bfs_seq = bfs_order(G, START)
dfs_seq = dfs_order(G, START)
print("BFS order:", bfs_seq)
print("DFS order:", dfs_seq)

# ---------------------------
# 5) Load India map
# ---------------------------
world = gpd.read_file(MAP_FILE)
# Depending on the geojson properties, the country name column can differ.
# nvkelso's geojson uses "ADMIN" for country name.
country_col = None
for col in ["ADMIN", "name", "NAME"]:
    if col in world.columns:
        country_col = col
        break
if country_col is None:
    raise RuntimeError("Could not find country name column in geojson.")
india = world[world[country_col] == "India"]

# ---------------------------
# 6) Prepare plotting
# ---------------------------
# Node positions as (lon, lat)
pos = {n: (data['lon'], data['lat']) for n, data in G.nodes(data=True)}

# figure
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
india.plot(ax=ax, color="lightgrey", edgecolor="black")

# Draw all nodes (cities) as background points
lons = [pos[n][0] for n in G.nodes()]
lats = [pos[n][1] for n in G.nodes()]
ax.scatter(lons, lats, s=40, zorder=5)
# annotate city names lightly
for n in G.nodes():
    x, y = pos[n]
    ax.text(x + 0.25, y + 0.25, n, fontsize=8, zorder=6)

ax.set_title("City traversal (BFS → pause → DFS)")
ax.set_xlim(65, 95)   # India-ish bounding box
ax.set_ylim(6, 36)

# We'll draw traversal lines incrementally onto two collections:
bfs_lines = []  # list of (xlist, ylist) for BFS segments
dfs_lines = []  # same for DFS

# helper to create segment list from an order
def segments_from_order(order):
    segs = []
    for i in range(len(order) - 1):
        c1 = order[i]
        c2 = order[i+1]
        x1, y1 = pos[c1]
        x2, y2 = pos[c2]
        segs.append(((x1, x2), (y1, y2)))
    return segs

bfs_segs = segments_from_order(bfs_seq)
dfs_segs = segments_from_order(dfs_seq)

# animation parameters
pause_frames = 12  # pause between BFS and DFS (frames)
fps = 4
total_frames = len(bfs_segs) + pause_frames + len(dfs_segs)

# current artists we will update
line_artists = []    # artists drawn (so we can animate their visibility)
point_artists = []   # small circle marker at the newly visited city

# initialize: empty lists (no traversal lines)
def init():
    # nothing to draw yet except base map & city points/names already drawn
    return []

# animation update function
def update(frame):
    """
    frame goes 0 .. total_frames-1
    0..len(bfs_segs)-1  -> draw BFS segments incrementally
    next pause_frames   -> hold BFS final state
    remaining frames    -> draw DFS segments incrementally (overwrite lines)
    """
    # clear previous traversal artists
    for art in line_artists + point_artists:
        try:
            art.remove()
        except Exception:
            pass
    line_artists.clear()
    point_artists.clear()

    if frame < len(bfs_segs):
        # drawing BFS: show segments 0..frame
        segs_to_draw = bfs_segs[:frame+1]
        # draw lines
        for (xs, ys) in segs_to_draw:
            ln, = ax.plot([xs[0], xs[1]], [ys[0], ys[1]], linewidth=2.2, alpha=0.85, zorder=7)
            line_artists.append(ln)
        # highlight the last visited city
        last_city = bfs_seq[frame+1] if frame+1 < len(bfs_seq) else bfs_seq[-1]
        x, y = pos[last_city]
        pt = ax.scatter([x], [y], s=120, edgecolors='black', linewidth=0.8, zorder=9)
        point_artists.append(pt)
    elif frame < len(bfs_segs) + pause_frames:
        # pause: show full BFS traversal
        for (xs, ys) in bfs_segs:
            ln, = ax.plot([xs[0], xs[1]], [ys[0], ys[1]], linewidth=2.2, alpha=0.85, zorder=7)
            line_artists.append(ln)
        # mark final BFS node
        last_city = bfs_seq[-1]
        x, y = pos[last_city]
        pt = ax.scatter([x], [y], s=120, edgecolors='black', linewidth=0.8, zorder=9)
        point_artists.append(pt)
    else:
        # DFS drawing phase: compute local frame index
        dfs_frame = frame - len(bfs_segs) - pause_frames
        segs_to_draw = dfs_segs[:dfs_frame+1]
        for (xs, ys) in segs_to_draw:
            ln, = ax.plot([xs[0], xs[1]], [ys[0], ys[1]], linewidth=2.2, linestyle='--', alpha=0.9, zorder=7)
            line_artists.append(ln)
        # highlight last visited in DFS
        last_city = dfs_seq[dfs_frame+1] if dfs_frame+1 < len(dfs_seq) else dfs_seq[-1]
        x, y = pos[last_city]
        pt = ax.scatter([x], [y], s=120, edgecolors='black', linewidth=0.8, zorder=9)
        point_artists.append(pt)

    # return artists that were changed (required by FuncAnimation)
    return line_artists + point_artists

# ---------------------------
# 7) Build and save animation (MP4)
# ---------------------------
anim = animation.FuncAnimation(
    fig, update, init_func=init, frames=total_frames, interval=1000/fps, blit=False
)

out_file = "traversal.mp4"
print(f"Saving animation to {out_file} ... (this may take a few seconds)")
# Use FFMpegWriter via save
Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='gagan'), bitrate=1800)
anim.save(out_file, writer=writer)
print("Saved:", out_file)

# optionally show on screen (commented out)
plt.show()

