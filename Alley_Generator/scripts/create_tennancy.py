import bpy
import bmesh
import math
from mathutils import Vector

# ==============================================================================
# SECTION 1: CONSTANTS & SETTINGS
# ==============================================================================
PLANE_SIZE      = 2.0
PLANE_LOCATION  = (0, 0, 0)
PLANE_ROTATION  = (math.radians(90), 0, 0)

MARGIN_LEFT     = 0.4
MARGIN_RIGHT    = 0.4
MARGIN_TOP      = 0.2
MARGIN_BOTTOM   = 0.2

EXTRUDE_SHORT   = 0.03
EXTRUDE_LONG    = 0.05

# Material names and colors
MAT_NAME_WINDOW_FACE = "Mat_Window_Face"
MAT_NAME_MARGIN_LONG = "Mat_Window_Margin_Long"
MAT_NAME_MARGIN_SHORT = "Mat_Window_Margin_Short"

COLOR_WINDOW_FACE = (1.0, 1.0, 1.0, 1.0)   # White
COLOR_MARGIN_LONG = (1.0, 0.0, 0.0, 1.0)    # Red
COLOR_MARGIN_SHORT = (0.0, 0.0, 1.0, 1.0)   # Blue

# ==============================================================================
# SECTION 2: HELPER FUNCTIONS
# ==============================================================================
def get_or_create_material(mat_name, color):
    """Retrieve an existing material or create a new one with the given name and base color."""
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                node.inputs["Base Color"].default_value = color
                break
    return mat

def extrude_faces_by_material(obj, material_index, extrude_amount):
    """Extrude all faces on the given object that match the specified material index."""
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(obj.data)
    for face in bm.faces:
        if face.material_index == material_index:
            face.select = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.extrude_faces_move()
    bpy.ops.transform.shrink_fatten(value=extrude_amount, use_even_offset=True)
    bpy.ops.object.mode_set(mode='OBJECT')

# ==============================================================================
# SECTION 3: MAIN SCRIPT (Create Grid with Extruded Faces)
# ==============================================================================
# 3.1: Clear Scene & Create Base Plane
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add(
    size=PLANE_SIZE,
    location=PLANE_LOCATION,
    rotation=PLANE_ROTATION
)
plane_obj = bpy.context.object
plane_obj.name = "Simple_Plane"

# 3.2: Set Up Materials
mat_window_face = get_or_create_material(MAT_NAME_WINDOW_FACE, COLOR_WINDOW_FACE)
mat_margin_long = get_or_create_material(MAT_NAME_MARGIN_LONG, COLOR_MARGIN_LONG)
mat_margin_short = get_or_create_material(MAT_NAME_MARGIN_SHORT, COLOR_MARGIN_SHORT)

if MAT_NAME_WINDOW_FACE not in [m.name for m in plane_obj.data.materials]:
    plane_obj.data.materials.append(mat_window_face)
if MAT_NAME_MARGIN_LONG not in [m.name for m in plane_obj.data.materials]:
    plane_obj.data.materials.append(mat_margin_long)
if MAT_NAME_MARGIN_SHORT not in [m.name for m in plane_obj.data.materials]:
    plane_obj.data.materials.append(mat_margin_short)

idx_window_face = plane_obj.data.materials.find(MAT_NAME_WINDOW_FACE)
idx_margin_long = plane_obj.data.materials.find(MAT_NAME_MARGIN_LONG)
idx_margin_short = plane_obj.data.materials.find(MAT_NAME_MARGIN_SHORT)

# 3.3: Compute Cut Positions (Local Coordinates)
x_left_default   = -PLANE_SIZE / 2 + PLANE_SIZE / 3
x_right_default  =  PLANE_SIZE / 2 - PLANE_SIZE / 3
y_bottom_default = -PLANE_SIZE / 2 + PLANE_SIZE / 3
y_top_default    =  PLANE_SIZE / 2 - PLANE_SIZE / 3

x_left_cut   = (-PLANE_SIZE / 2) + MARGIN_LEFT   * (x_left_default - (-PLANE_SIZE / 2))
x_right_cut  = ( PLANE_SIZE / 2) - MARGIN_RIGHT  * ((PLANE_SIZE / 2) - x_right_default)
y_bottom_cut = (-PLANE_SIZE / 2) + MARGIN_BOTTOM * (y_bottom_default - (-PLANE_SIZE / 2))
y_top_cut    = (PLANE_SIZE / 2) - MARGIN_TOP    * ((PLANE_SIZE / 2) - y_top_default)

# 3.4: Create 3×3 Grid via Bisections
mesh = plane_obj.data
bm = bmesh.new()
bm.from_mesh(mesh)

# Vertical bisections
geom_all = bm.faces[:] + bm.edges[:] + bm.verts[:]
bmesh.ops.bisect_plane(
    bm,
    geom=geom_all,
    plane_co=(x_left_cut, 0, 0),
    plane_no=(1, 0, 0),
    clear_inner=False,
    clear_outer=False
)
geom_all = bm.faces[:] + bm.edges[:] + bm.verts[:]
bmesh.ops.bisect_plane(
    bm,
    geom=geom_all,
    plane_co=(x_right_cut, 0, 0),
    plane_no=(1, 0, 0),
    clear_inner=False,
    clear_outer=False
)

# Horizontal bisections
geom_all = bm.faces[:] + bm.edges[:] + bm.verts[:]
bmesh.ops.bisect_plane(
    bm,
    geom=geom_all,
    plane_co=(0, y_bottom_cut, 0),
    plane_no=(0, 1, 0),
    clear_inner=False,
    clear_outer=False
)
geom_all = bm.faces[:] + bm.edges[:] + bm.verts[:]
bmesh.ops.bisect_plane(
    bm,
    geom=geom_all,
    plane_co=(0, y_top_cut, 0),
    plane_no=(0, 1, 0),
    clear_inner=False,
    clear_outer=False
)

bm.to_mesh(mesh)
bm.free()

# 3.5: Assign Materials to Grid Faces
x_edge_left   = -PLANE_SIZE / 2
x_edge_right  =  PLANE_SIZE / 2
y_edge_bottom = -PLANE_SIZE / 2
y_edge_top    =  PLANE_SIZE / 2

left_boundary   = (x_edge_left + x_left_cut) / 2
right_boundary  = (x_right_cut + x_edge_right) / 2
bottom_boundary = (y_edge_bottom + y_bottom_cut) / 2
top_boundary    = (y_top_cut + y_edge_top) / 2

left_margin   = x_left_cut - x_edge_left
right_margin  = x_edge_right - x_right_cut
total_horizontal_margin = left_margin + right_margin
bottom_margin = y_bottom_cut - y_edge_bottom
top_margin    = y_edge_top - y_top_cut
total_vertical_margin = bottom_margin + top_margin

horizontal_is_long = total_horizontal_margin >= total_vertical_margin

for poly in plane_obj.data.polygons:
    cx, cy, _ = poly.center
    if abs(cx) < 0.1 and abs(cy) < 0.1:
        poly.material_index = idx_window_face
    else:
        if horizontal_is_long:
            if cy < bottom_boundary or cy > top_boundary:
                poly.material_index = idx_margin_long
            elif cx < left_boundary or cx > right_boundary:
                poly.material_index = idx_margin_short
            else:
                poly.material_index = idx_margin_long
        else:
            if cx < left_boundary or cx > right_boundary:
                poly.material_index = idx_margin_long
            elif cy < bottom_boundary or cy > top_boundary:
                poly.material_index = idx_margin_short
            else:
                poly.material_index = idx_margin_long

# 3.6: Extrude Margin Faces (Leave Center Face Untouched)
effective_extrude_short = PLANE_SIZE * EXTRUDE_SHORT
effective_extrude_long  = PLANE_SIZE * EXTRUDE_LONG

extrude_faces_by_material(plane_obj, idx_margin_long, effective_extrude_long)
extrude_faces_by_material(plane_obj, idx_margin_short, effective_extrude_short)

print("Grid created and materials assigned with differentiated colors.")
print("Extrusion of margin faces complete (center face left untouched).")

# ==============================================================================
# SECTION 4: ADD NEW TENNANCY OBJECT(S)
# ==============================================================================
try:
    tennancy = bpy.data.texts["tennancy.py"].as_module()
except KeyError:
    print("Error: 'tennancy.py' text block not found. Tennancy creation aborted.")
else:
    # Example: Create a new tennancy window at a specified location
    new_tennancy = tennancy.create_blank_tennancy(
        plane_size=2.0,
        plane_location=(0, 0, 0),  # Adjust location as needed
        plane_rotation=(math.radians(90), 0, 0),  # Adjust rotation as needed
        margin_left=MARGIN_LEFT,
        margin_right=MARGIN_RIGHT,
        margin_top=MARGIN_TOP,
        margin_bottom=MARGIN_BOTTOM,
        extrude_short=EXTRUDE_SHORT,
        extrude_long=EXTRUDE_LONG
    )
    print("New tennancy added:", new_tennancy)
