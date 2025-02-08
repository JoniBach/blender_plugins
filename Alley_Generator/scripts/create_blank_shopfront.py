import bpy
import bmesh
import math
from mathutils import Vector

# ==============================================================================
# SECTION 1: CONSTANTS & SETTINGS
# ==============================================================================

# --- Plane Setup ---
# This is now the master value: when you change PLANE_SIZE, everything else scales.
PLANE_SIZE          = 1  # Change this value to scale the entire component.
PLANE_LOCATION      = (0, 0, 0)
PLANE_ORIENTATION   = 90

# --- Base Dimensions (original) ---
SIGN_HEIGHT         = 0.25
SIGN_DEPTH          = 0.1 
SIGN_FACE_DEPTH     = -0.025
SIGN_BORDER_MARGIN  = 0.01

PILLAR_WIDTH_LEFT   = 0.4
PILLAR_WIDTH_RIGHT  = 0.4
PILLAR_DEPTH        = 0.05 

FRONT_FACE_DEPTH    = -0.05

SHUTTER_SEGMENTS    = 11
SHUTTER_DEPTH       = 0.005
SHUTTER_CLOSED      = 0.15 

# Compute a scale factor relative to the original (base) size of 2.0.
scale = PLANE_SIZE / 1.0

# --- Base Dimensions (scaled) ---
SCALED_HORIZONTAL_LOOP_Y        = SIGN_HEIGHT  * scale
SCALED_VERTICAL_LOOP_X_LEFT     = -PILLAR_WIDTH_LEFT * scale
SCALED_VERTICAL_LOOP_X_RIGHT    =  PILLAR_WIDTH_RIGHT * scale

# --- Extrusion Parameters (scaled) ---
SCALED_SIGN_DEPTH               = SIGN_DEPTH   * scale
SCALED_SIGN_FACE_DEPTH          = SIGN_FACE_DEPTH  * scale
SCALED_SIGN_BORDER_MARGIN       = SIGN_BORDER_MARGIN  * scale

SCALED_PILLAR_DEPTH             = PILLAR_DEPTH  * scale
SCALED_FRONT_FACE_DEPTH         = FRONT_FACE_DEPTH * scale
SCALED_SHUTTER_SEGMENTS         = SHUTTER_SEGMENTS
SCALED_SHUTTER_DEPTH            = SHUTTER_DEPTH * scale
SCALED_SHUTTER_CLOSED           = SHUTTER_CLOSED  * scale

# --- Internal/Factory Parameters (scaled where applicable) ---
PLANE_ROTATION                  = (math.radians(90), 0, math.radians(PLANE_ORIENTATION))
SHUTTER_EXTRUDE_DISTANCE        = -1.6 * scale
SIDE_EXTRUDE_TRANSLATION        = (0, -0.1 * scale, 0)
SHUTTER_BISECT_TOLERANCE        = 0.001 * scale
SHUTTER_EDGE_TOLERANCE          = 0.01  * scale
SHUTTER_NUM_CUTS                = (SCALED_SHUTTER_SEGMENTS * 2) - 1
SIDE_Y_THRESHOLD                = SCALED_HORIZONTAL_LOOP_Y
SIDE_X_THRESHOLD                = abs(SCALED_VERTICAL_LOOP_X_LEFT) * 5.0 / 8.0
FRONT_FACE_MIN_Y                = SCALED_HORIZONTAL_LOOP_Y * 0.5
FRONT_FACE_X_RANGE              = SIDE_X_THRESHOLD
TOP_BAR_ASSIGN_THRESHOLD        = SCALED_HORIZONTAL_LOOP_Y - 0.05 * scale

# --- Material Colors (unchanged) ---
COLOR_MAT_SIGN       = (1.0, 0.2, 0.2, 1)
COLOR_MAT_COLUMN     = (0.2, 1.0, 0.2, 1)
COLOR_MAT_FRONT      = (0.2, 0.2, 1.0, 1)
COLOR_MAT_SIGN_FACE  = (1.0, 0.4, 0.4, 1)
COLOR_SHUTTER        = (0.8, 0.8, 0.2, 1)
COLOR_MAT_FRONT_FACE = (0.4, 0.4, 1.0, 1)

# ==============================================================================
# SECTION 2: HELPER FUNCTIONS
# ==============================================================================

def get_or_create_material(mat_name, color):
    """Get or create a material with the given name and base color."""
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
        mat.diffuse_color = color
    return mat

def compute_cut_positions(shutter_faces):
    """Compute and return the Y positions for cuts given shutter faces."""
    shutter_verts = {v for face in shutter_faces for v in face.verts}
    y_coords = [v.co.y for v in shutter_verts]
    min_y, max_y = min(y_coords), max(y_coords)
    return [min_y + (i + 1) * (max_y - min_y) / (SHUTTER_NUM_CUTS + 1)
            for i in range(SHUTTER_NUM_CUTS)]

# ==============================================================================
# SECTION 3: MAIN SCRIPT
# ==============================================================================

# ------------------------------------------------------------------------------
# 3.1: Create Base Plane
# ------------------------------------------------------------------------------
bpy.ops.mesh.primitive_plane_add(
    size=PLANE_SIZE,
    enter_editmode=False,
    align='WORLD',
    location=PLANE_LOCATION,
    rotation=PLANE_ROTATION
)
plane_obj = bpy.context.active_object

# ------------------------------------------------------------------------------
# 3.2: Bisect the Plane to Form Loops
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane_obj.data)

# Bisect horizontally along Y = SCALED_HORIZONTAL_LOOP_Y
geom_all = list(bm.verts) + list(bm.edges) + list(bm.faces)
bmesh.ops.bisect_plane(
    bm,
    geom=geom_all,
    plane_co=(0, SCALED_HORIZONTAL_LOOP_Y, 0),
    plane_no=(0, 1, 0),
    dist=SHUTTER_BISECT_TOLERANCE,
    use_snap_center=False,
    clear_inner=False,
    clear_outer=False
)

# Bisect vertically at the left and right positions
for x_val in [SCALED_VERTICAL_LOOP_X_LEFT, SCALED_VERTICAL_LOOP_X_RIGHT]:
    geom_all = list(bm.verts) + list(bm.edges) + list(bm.faces)
    plane_no = (1, 0, 0) if x_val < 0 else (-1, 0, 0)
    bmesh.ops.bisect_plane(
        bm,
        geom=geom_all,
        plane_co=(x_val, 0, 0),
        plane_no=plane_no,
        dist=SHUTTER_BISECT_TOLERANCE,
        use_snap_center=False,
        clear_inner=False,
        clear_outer=False
    )
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.3: Create & Assign Materials
# ------------------------------------------------------------------------------
# Create materials (or retrieve if they already exist)
mat_sign       = get_or_create_material("Mat_Sign", COLOR_MAT_SIGN)
mat_column     = get_or_create_material("Mat_Column", COLOR_MAT_COLUMN)
mat_front      = get_or_create_material("Mat_Front", COLOR_MAT_FRONT)
mat_sign_face  = get_or_create_material("Mat_Sign_face", COLOR_MAT_SIGN_FACE)
mat_front_face = get_or_create_material("Mat_Front_face", COLOR_MAT_FRONT_FACE)
mat_shutter    = get_or_create_material("Shutter", COLOR_SHUTTER)

# Assign materials to the mesh in a specific order (material indices matter)
mesh = plane_obj.data
mesh.materials.clear()
mesh.materials.append(mat_sign)       # index 0
mesh.materials.append(mat_column)     # index 1
mesh.materials.append(mat_front)      # index 2
mesh.materials.append(mat_sign_face)  # index 3
mesh.materials.append(mat_front_face) # index 4
mesh.materials.append(mat_shutter)    # index 5

# ------------------------------------------------------------------------------
# 3.4: Clean Up Mesh - Remove Extra Faces
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane_obj.data)
for face in bm.faces:
    if face.calc_center_median().y > SCALED_HORIZONTAL_LOOP_Y:
        face.select = True
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.mesh.dissolve_faces()
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.5: Assign Face Materials Based on Position
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane_obj.data)
for face in bm.faces:
    center = face.calc_center_median()
    if center.y >= TOP_BAR_ASSIGN_THRESHOLD:
        face.material_index = 0  # Top bar: Mat_Sign
    elif center.x < -SIDE_X_THRESHOLD or center.x > SIDE_X_THRESHOLD:
        face.material_index = 1  # Sides: Mat_Column
    else:
        face.material_index = 2  # Front: Mat_Front
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.6: Extrude the Top Bar
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane_obj.data)
for face in bm.faces:
    face.select = (face.material_index == 0)
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.mesh.extrude_faces_move()
bpy.ops.transform.shrink_fatten(value=SCALED_SIGN_DEPTH, use_even_offset=True)
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.7: Extrude the Side Columns (Mat_Column)
# ------------------------------------------------------------------------------
# Find the Mat_Column material index.
column_mat_index = None
for i, mat in enumerate(mesh.materials):
    if mat.name == "Mat_Column":
        column_mat_index = i
        break
if column_mat_index is None:
    raise ValueError("Material 'Mat_Column' not found on the object!")

bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane_obj.data)
bpy.ops.mesh.select_all(action='DESELECT')
for face in bm.faces:
    if face.material_index == column_mat_index:
        face.select = True
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.mesh.extrude_faces_move()
bpy.ops.transform.shrink_fatten(value=SCALED_PILLAR_DEPTH, use_even_offset=True)
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.8: Inset and Extrude the Front Sign Face
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane_obj.data)
bpy.ops.mesh.select_all(action='DESELECT')

# Identify candidate face(s) for the front sign area based on position & normal.
candidates = [face for face in bm.faces
              if (-FRONT_FACE_X_RANGE <= face.calc_center_median().x <= FRONT_FACE_X_RANGE and
                  face.calc_center_median().y > FRONT_FACE_MIN_Y and
                  abs(face.normal.y) < 0.3)]
if not candidates:
    raise RuntimeError("No candidate front sign faces found!")
front_face = min(candidates, key=lambda f: f.calc_center_median().y)
front_face.select = True
bmesh.update_edit_mesh(plane_obj.data)

# Inset the face and then extrude it.
bpy.ops.mesh.inset(thickness=SCALED_SIGN_BORDER_MARGIN)
bpy.ops.mesh.extrude_faces_move()
bpy.ops.transform.shrink_fatten(value=SCALED_SIGN_FACE_DEPTH, use_even_offset=True)

# Assign the resulting face the Mat_Sign_face material.
bm = bmesh.from_edit_mesh(plane_obj.data)
for face in bm.faces:
    if face.select:
        face.material_index = 3  # Mat_Sign_face
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.9: Extrude the Front Face (Mat_Front)
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane_obj.data)
bpy.ops.mesh.select_all(action='DESELECT')
for face in bm.faces:
    if face.material_index == 2:
        face.select = True
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.mesh.extrude_faces_indiv()
bpy.ops.transform.shrink_fatten(value=SCALED_FRONT_FACE_DEPTH, use_even_offset=True)

# Reassign extruded faces to Mat_Front_face.
bm = bmesh.from_edit_mesh(plane_obj.data)
for face in bm.faces:
    if face.select:
        face.material_index = 4  # Mat_Front_face
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.10: Create the "RedBlue_Interface" Vertex Group
# ------------------------------------------------------------------------------
v2mats = {v.index: set() for v in plane_obj.data.vertices}
for poly in plane_obj.data.polygons:
    m = poly.material_index
    for vid in poly.vertices:
        v2mats[vid].add(m)
interface_verts = [vid for vid, mats in v2mats.items() if 0 in mats and 2 in mats]

vg = plane_obj.vertex_groups.get("RedBlue_Interface")
if vg is None:
    vg = plane_obj.vertex_groups.new(name="RedBlue_Interface")
vg.add(interface_verts, 1.0, 'ADD')
print("Created vertex group 'RedBlue_Interface' with", len(interface_verts), "vertices.")

# ------------------------------------------------------------------------------
# 3.11: Extrude the Shutter Region
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane_obj.data)
bpy.ops.mesh.select_mode(type="EDGE")
bpy.ops.mesh.select_all(action='DESELECT')

# Get interface vertex indices from the RedBlue_Interface group.
interface_indices = []
for v in plane_obj.data.vertices:
    for g in v.groups:
        if g.group == vg.index and g.weight > 0:
            interface_indices.append(v.index)
            break

# Select edges connecting two interface vertices (with nearly identical Z values).
for edge in bm.edges:
    edge.select = False
    if (edge.verts[0].index in interface_indices and edge.verts[1].index in interface_indices):
        if abs(edge.verts[0].co.z - edge.verts[1].co.z) < SHUTTER_BISECT_TOLERANCE:
            edge.select = True
bmesh.update_edit_mesh(plane_obj.data)

# Extrude the selected region to form the shutter.
bm = bmesh.from_edit_mesh(plane_obj.data)
old_face_indices = {f.index for f in bm.faces}
shutter_distance = abs(SHUTTER_EXTRUDE_DISTANCE) * abs(SCALED_SHUTTER_CLOSED)
bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0, 0, -shutter_distance)})
bmesh.update_edit_mesh(plane_obj.data)
bm = bmesh.from_edit_mesh(plane_obj.data)
new_faces = [f for f in bm.faces if f.index not in old_face_indices]
for face in new_faces:
    face.material_index = 5  # Shutter material
bmesh.update_edit_mesh(plane_obj.data)
bpy.ops.mesh.select_mode(type="FACE")
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.12: Create Horizontal Cuts & Offset Shutter Edges
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(mesh)

# Identify shutter material index.
shutter_mat_index = None
for i, mat in enumerate(mesh.materials):
    if mat.name == "Shutter":
        shutter_mat_index = i
        break
if shutter_mat_index is None:
    raise ValueError("Shutter material not found on the object!")

# Get current shutter faces and compute cut positions.
shutter_faces = [face for face in bm.faces if face.material_index == shutter_mat_index]
if not shutter_faces:
    raise ValueError("No shutter faces found!")
cut_positions = compute_cut_positions(shutter_faces)

# For each computed Y-position, bisect the shutter geometry.
for y in cut_positions:
    shutter_faces = [face for face in bm.faces if face.material_index == shutter_mat_index]
    shutter_edges = {edge for face in shutter_faces for edge in face.edges}
    shutter_verts = {v for face in shutter_faces for v in face.verts}
    geom = list(shutter_faces) + list(shutter_edges) + list(shutter_verts)
    bmesh.ops.bisect_plane(
        bm,
        geom=geom,
        plane_co=(0, y, 0),
        plane_no=(0, 1, 0),
        dist=SHUTTER_BISECT_TOLERANCE,
        use_snap_center=False,
        clear_inner=False,
        clear_outer=False
    )
bmesh.update_edit_mesh(mesh)
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------------------
# 3.13: Select Specific Shutter Edges and Offset Them
# ------------------------------------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(mesh)

# Recompute shutter faces and cut positions after bisecting.
shutter_faces = [face for face in bm.faces if face.material_index == shutter_mat_index]
cut_positions = compute_cut_positions(shutter_faces)
cut_edges = {}
for edge in bm.edges:
    if not any(face.material_index == shutter_mat_index for face in edge.link_faces):
        continue
    if abs(edge.verts[0].co.y - edge.verts[1].co.y) < SHUTTER_EDGE_TOLERANCE:
        median_y = (edge.verts[0].co.y + edge.verts[1].co.y) / 2.0
        for idx, cp in enumerate(cut_positions):
            if abs(median_y - cp) < SHUTTER_EDGE_TOLERANCE:
                if idx not in cut_edges:
                    cut_edges[idx] = edge
                break

# Select every other edge from the computed cuts.
for idx, edge in cut_edges.items():
    edge.select = (idx % 2 == 0)
bmesh.update_edit_mesh(mesh)

# Offset the selected shutter edge vertices along Z.
selected_verts = {v for edge in bm.edges if edge.select for v in edge.verts}
for v in selected_verts:
    v.co.z += SCALED_SHUTTER_DEPTH
bmesh.update_edit_mesh(mesh)
bpy.ops.object.mode_set(mode='OBJECT')

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
print("Script completed successfully.")
