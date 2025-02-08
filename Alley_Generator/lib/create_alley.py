import bpy
import bmesh
import random
import colorsys
from typing import Tuple

# ------------------ User Parameters ------------------

MERGE_THRESHOLD = 0.0001      # Distance within which vertices will be merged
RANDOM_SEED = 1245            # Seed for all random operations

# Building cell parameters
NUM_BUILDINGS = 10            # Number of building cells
MIN_STORIES = 3               # Minimum extrusions (stories) per cell
MAX_STORIES = 8               # Maximum extrusions (stories) per cell
EXTRUSION_HEIGHT = 2.0        # Height of each floor
MIN_BUILDING_WIDTH = 1.5      # Minimum building width (X-axis)
MAX_BUILDING_WIDTH = 3.0      # Maximum building width (X-axis)
BUILDING_DEPTH = 2.0          # Fixed depth (Y-axis)

# Wire loom parameters
WIRE_STORY_COUNT = 1          # Number of stories (above ground floor) that get wires
WIRE_BUNDLE_COUNT = 3         # Number of wires (bundles) per story
BASE_SAG = 0.3                # Base sag for wires
SAG_RANDOMNESS = 0.2          # Maximum random variation in sag
BASE_OFFSET = 0.5             # Base offset along Y-axis for endpoints
OFFSET_RANDOMNESS = 0.1       # Additional random offset per endpoint
LEFT_LOOM_RANDOMNESS = 0.2    # Random offset range for left wire bundle
RIGHT_LOOM_RANDOMNESS = 0.2   # Random offset range for right wire bundle
EXTRUDE_AMOUNT = 0            # Amount for extruding the bottom faces (if needed)
GROUND_FLOOR_SEPERATION = 1.2 # Scaling factor for the ground floor walls

# Spacing between cells (arranged gapless along Y)
SPACING = BUILDING_DEPTH

# --- New Parameter: Which element(s) to select at the end ---
# Options: "none", "floor", "ceiling", "ground", "story", "joining", "wire", "all_building", "all"
FINAL_SELECTION = "floor"

# -----------------------------------------------------
# Type alias for a 3D coordinate
Vec3 = Tuple[float, float, float]

# ======================================================
# HELPER FUNCTION: HSV to RGBA
# ======================================================

def hsv_to_rgba(h: float, s: float, v: float, a: float = 1.0) -> Tuple[float, float, float, float]:
    """Convert an HSV value (with h in [0,1]) to an RGBA tuple."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, a)

# ======================================================
# MATERIALS & MARKING FUNCTIONS
# ======================================================

def get_or_create_material(name: str, color: Tuple[float, float, float, float]) -> bpy.types.Material:
    """Gets or creates a material with the given name and diffuse color."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name=name)
        # For Blender 2.8+ the diffuse_color is an RGBA tuple.
        mat.diffuse_color = color
        return mat

def create_materials() -> dict:
    """
    Create (or get) a set of materials used to mark elements.
      - 'floor': the horizontal floor face (violet).
      - 'ceiling': horizontal faces above the floor (orange).
      - 'ground': vertical walls on the ground floor (red).
      - 'story': vertical walls on the upper stories (green).
      - 'joining': the joining walls (blue).
      - 'wire': the wires (yellow).
    """
    mats = {
        "floor": get_or_create_material("Mat_Floor", hsv_to_rgba(0.75, 1, 1)),      # Violet
        "ceiling": get_or_create_material("Mat_Ceiling", hsv_to_rgba(0.08, 1, 1)),   # Orange
        "ground": get_or_create_material("Mat_GroundWall", hsv_to_rgba(0.0, 1, 1)),   # Red
        "story": get_or_create_material("Mat_StoryWall", hsv_to_rgba(0.33, 1, 1)),    # Green
        "joining": get_or_create_material("Mat_JoiningWall", hsv_to_rgba(0.66, 1, 1)),# Blue
        "wire": get_or_create_material("Mat_Wire", hsv_to_rgba(0.1667, 1, 1))           # Yellow
    }
    return mats

def assign_face_materials(obj: bpy.types.Object, extrusion_height: float, mats: dict) -> None:
    """
    Iterates over the faces of the mesh in 'obj' (the joined building and joining walls)
    and assigns a material index based on simple criteria:
      - Nearly horizontal faces with center at z ≈ 0 are marked as floor.
      - All other horizontal faces are marked as ceiling.
      - Vertical faces with a strong X component:
          * If their center lies below roughly one floor’s height, they are marked as ground-floor walls.
          * Otherwise they are marked as story walls.
      - Vertical faces with a dominant Y component are marked as joining walls.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # Clear any existing materials and add our five materials in order:
    # Index 0: Floor, 1: Ground Wall, 2: Story Wall, 3: Joining Wall, 4: Ceiling
    mesh.materials.clear()
    mesh.materials.append(mats["floor"])     # 0
    mesh.materials.append(mats["ground"])      # 1
    mesh.materials.append(mats["story"])       # 2
    mesh.materials.append(mats["joining"])     # 3
    mesh.materials.append(mats["ceiling"])     # 4

    for face in bm.faces:
        center = face.calc_center_median()
        normal = face.normal
        
        # If the face is nearly horizontal...
        if abs(normal.z) > 0.9:
            if center.z < 0.1:
                face.material_index = 0  # Floor
            else:
                face.material_index = 4  # Ceiling
        else:
            # For vertical faces, check the dominant axis of the normal.
            if abs(normal.x) > 0.7:
                # If near the ground floor, mark as ground wall; else, story wall.
                if center.z < (extrusion_height + 0.1):
                    face.material_index = 1
                else:
                    face.material_index = 2
            elif abs(normal.y) > 0.7:
                face.material_index = 3
            else:
                face.material_index = 1  # Fallback
            
    bm.to_mesh(mesh)
    bm.free()

def assign_wire_material(curve_obj: bpy.types.Object, mats: dict) -> None:
    """
    Assigns the wire material to the given curve object.
    """
    curve = curve_obj.data
    curve.materials.clear()
    curve.materials.append(mats["wire"])

# ======================================================
# FINAL SELECTION FUNCTION
# ======================================================

def select_final_elements(final_building_obj: bpy.types.Object, wire_objects: list, selection: str) -> None:
    """
    Based on the 'selection' parameter, selects the corresponding elements:
      - "floor", "ceiling", "ground", "story", "joining" – select faces (by material index) in the building mesh.
      - "all_building" – select all faces in the building mesh.
      - "wire" – select all wire (curve) objects.
      - "all" – select all objects.
      - "none" – do nothing.
    """
    selection = selection.lower()
    if selection == "none":
        return

    # Deselect everything first.
    bpy.ops.object.select_all(action='DESELECT')
    
    if selection == "all":
        bpy.ops.object.select_all(action='SELECT')
        return

    # Building faces selection.
    building_categories = {
        "floor": 0,
        "ground": 1,
        "story": 2,
        "joining": 3,
        "ceiling": 4,
        "all_building": -1
    }
    if selection in building_categories:
        bpy.context.view_layer.objects.active = final_building_obj
        final_building_obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        if selection == "all_building":
            bpy.ops.mesh.select_all(action='SELECT')
        else:
            target_index = building_categories[selection]
            mesh = final_building_obj.data
            bm = bmesh.from_edit_mesh(mesh)
            for face in bm.faces:
                if face.material_index == target_index:
                    face.select = True
            bmesh.update_edit_mesh(mesh, destructive=True)
        bpy.ops.object.mode_set(mode='OBJECT')
    elif selection == "wire":
        # Select all wire objects.
        for w in wire_objects:
            w.select_set(True)
        if wire_objects:
            bpy.context.view_layer.objects.active = wire_objects[0]

# ======================================================
# GEOMETRY CREATION FUNCTIONS
# ======================================================

def cleanup_scene() -> None:
    """Deletes all objects in the current Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_extruded_building(
    num_extrusions: int,
    extrusion_height: float,
    location: Vec3,
    building_width: float,
    building_depth: float
) -> bpy.types.Object:
    """
    Creates a building cell by extruding a plane upward.
    The cell has a rectangular footprint (random width, fixed depth).
    Only the left/right edges are extruded; the front/back edges remain.
    """
    bpy.ops.mesh.primitive_plane_add(size=1, location=location)
    building_obj = bpy.context.active_object

    # Scale the plane to the desired footprint.
    building_obj.scale.x = building_width
    building_obj.scale.y = building_depth
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Go to Edit Mode and select left/right edges.
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action='DESELECT')
    mesh = building_obj.data
    bm = bmesh.from_edit_mesh(mesh)
    for edge in bm.edges:
        # Select left edges.
        if all(v.co.x < -0.9 * (building_width / 2) for v in edge.verts):
            edge.select = True
        # Select right edges.
        if all(v.co.x > 0.9 * (building_width / 2) for v in edge.verts):
            edge.select = True
    bmesh.update_edit_mesh(mesh)
    # Extrude the selected edges upward.
    bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, extrusion_height)})
    for _ in range(num_extrusions - 1):
        bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, extrusion_height)})
    bpy.ops.object.mode_set(mode='OBJECT')
    return building_obj

def add_wire_spline(curve_data: bpy.types.Curve, start: Vec3, end: Vec3, sag: float) -> None:
    """
    Adds a Bézier spline (with three control points) to the given curve data.
    """
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(2)
    bp0, bp1, bp2 = spline.bezier_points
    bp0.co = start
    bp2.co = end
    bp1.co = ( (start[0] + end[0]) / 2,
               (start[1] + end[1]) / 2,
               (start[2] + end[2]) / 2 - sag )
    for bp in spline.bezier_points:
        bp.handle_left_type = bp.handle_right_type = 'AUTO'

def create_wire_loom(
    num_extrusions: int,
    extrusion_height: float,
    wire_story_count: int,
    wire_bundle_count: int,
    base_sag: float,
    sag_randomness: float,
    base_offset: float,
    offset_randomness: float,
    left_loom_randomness: float,
    right_loom_randomness: float,
    building_location: Vec3,
    building_width: float
) -> bpy.types.Object:
    """
    Creates a wire loom for a building cell as a curve.
    """
    curve_data = bpy.data.curves.new(name="WireLoom", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.02
    curve_data.bevel_resolution = 4
    curve_data.resolution_u = 12

    max_story = min(wire_story_count, num_extrusions)
    for floor in range(1, max_story + 1):
        z = building_location[2] + floor * extrusion_height
        left_loom_offset = random.uniform(-left_loom_randomness, left_loom_randomness)
        right_loom_offset = random.uniform(-right_loom_randomness, right_loom_randomness)
        for bundle in range(wire_bundle_count):
            current_sag = base_sag + random.uniform(-sag_randomness, sag_randomness)
            left_individual = random.uniform(-offset_randomness, offset_randomness)
            right_individual = random.uniform(-offset_randomness, offset_randomness)
            left_y = base_offset + left_loom_offset + left_individual
            right_y = base_offset + right_loom_offset + right_individual
            start_point: Vec3 = (building_location[0] - building_width / 2,
                                  building_location[1] + left_y, z)
            end_point: Vec3 = (building_location[0] + building_width / 2,
                                building_location[1] + right_y, z)
            add_wire_spline(curve_data, start_point, end_point, current_sag)

    loom_obj = bpy.data.objects.new("WireLoom", curve_data)
    bpy.context.collection.objects.link(loom_obj)
    return loom_obj

def create_joining_wall_subdivided(x_left: float, x_right: float, y: float, height: float, subdivisions: int) -> bpy.types.Object:
    """
    Creates a joining wall as a vertically subdivided quad.
    """
    verts = []
    faces = []
    for i in range(subdivisions + 1):
        z = (height * i) / subdivisions
        verts.append((x_left, y, z))
        verts.append((x_right, y, z))
    for i in range(subdivisions):
        a = 2 * i
        b = a + 1
        c = 2 * (i + 1) + 1
        d = 2 * (i + 1)
        faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new("JoiningWallSubdivMesh")
    obj = bpy.data.objects.new("JoiningWallSubdiv", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return obj

def merge_vertices_by_distance(obj: bpy.types.Object, threshold: float) -> None:
    """
    Merges vertices in the given object's mesh that are within the given threshold.
    """
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=threshold)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

def select_bottom_extrusion_faces(obj: bpy.types.Object, extrusion_height: float, tol: float = 0.2) -> None:
    """
    Selects vertical faces belonging to the bottom (first) extrusion.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        if abs(face.normal.z) < 0.3:
            center_z = sum(v.co.z for v in face.verts) / len(face.verts)
            if tol < center_z < (extrusion_height - tol):
                face.select = True
    bmesh.update_edit_mesh(mesh)
    
    # Extrude the selected faces and scale them along the X-axis.
    bpy.ops.mesh.extrude_region_shrink_fatten(TRANSFORM_OT_shrink_fatten={"value": EXTRUDE_AMOUNT})
    bpy.ops.transform.resize(value=(GROUND_FLOOR_SEPERATION, 1, 1), constraint_axis=(True, False, False))
    bpy.ops.object.mode_set(mode='OBJECT')

def filter_ground_floor_faces(obj: bpy.types.Object, min_abs_x: float = 0.7) -> None:
    """
    Deselects any face that does not have a significant horizontal (X) component in its normal.
    (This helps to leave the joining walls unselected.)
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        if face.select:
            if abs(face.normal.x) < min_abs_x:
                face.select = False
    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')

# ======================================================
# MAIN FUNCTION
# ======================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    
    # Uncomment the following line to clear the scene:
    # cleanup_scene()

    # Create our marking materials.
    mats = create_materials()

    # ------------------ Build Parameters ------------------
    extrusion_height = EXTRUSION_HEIGHT
    min_building_width = MIN_BUILDING_WIDTH
    max_building_width = MAX_BUILDING_WIDTH   # Corrected variable name
    building_depth = BUILDING_DEPTH
    spacing = SPACING
    # -----------------------------------------------------

    building_objects = []
    wire_objects = []
    building_info = []  # For use when creating joining walls

    current_y = 0.0
    for i in range(NUM_BUILDINGS):
        building_width = random.uniform(min_building_width, max_building_width)
        num_extrusions = random.randint(MIN_STORIES, MAX_STORIES)
        building_height = (num_extrusions + 1) * extrusion_height
        building_location: Vec3 = (0, current_y, 0)
        # Create the building cell.
        b_obj = create_extruded_building(num_extrusions, extrusion_height,
                                         building_location, building_width, building_depth)
        building_objects.append(b_obj)
        building_info.append({
            'center_y': current_y,
            'width': building_width,
            'height': building_height,
            'extrusions': num_extrusions
        })
        # Create the wire loom.
        w_obj = create_wire_loom(num_extrusions, extrusion_height,
                                 WIRE_STORY_COUNT, WIRE_BUNDLE_COUNT,
                                 BASE_SAG, SAG_RANDOMNESS, BASE_OFFSET,
                                 OFFSET_RANDOMNESS, LEFT_LOOM_RANDOMNESS,
                                 RIGHT_LOOM_RANDOMNESS, building_location, building_width)
        wire_objects.append(w_obj)
        current_y += spacing

    # Join all building objects into one mesh.
    bpy.ops.object.select_all(action='DESELECT')
    for obj in building_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = building_objects[0]
    bpy.ops.object.join()
    joined_buildings = bpy.context.active_object

    # Create joining wall objects between adjacent cells.
    joining_wall_objects = []
    for i in range(len(building_info) - 1):
        cell1 = building_info[i]
        cell2 = building_info[i+1]
        # Use the visible wall height from the shorter cell (subtract one extrusion).
        if cell1['height'] <= cell2['height']:
            junction_y = cell1['center_y'] + building_depth / 2
            visible_height = cell1['height'] - extrusion_height
            subdivisions = cell1['extrusions']
        else:
            junction_y = cell2['center_y'] - building_depth / 2
            visible_height = cell2['height'] - extrusion_height
            subdivisions = cell2['extrusions']
        # Create joining wall for left edges.
        left_edge1 = -cell1['width'] / 2
        left_edge2 = -cell2['width'] / 2
        if abs(left_edge1 - left_edge2) > 1e-4:
            wall_obj = create_joining_wall_subdivided(min(left_edge1, left_edge2),
                                                      max(left_edge1, left_edge2),
                                                      junction_y, visible_height, subdivisions)
            joining_wall_objects.append(wall_obj)
        # Create joining wall for right edges.
        right_edge1 = cell1['width'] / 2
        right_edge2 = cell2['width'] / 2
        if abs(right_edge1 - right_edge2) > 1e-4:
            wall_obj = create_joining_wall_subdivided(min(right_edge1, right_edge2),
                                                      max(right_edge1, right_edge2),
                                                      junction_y, visible_height, subdivisions)
            joining_wall_objects.append(wall_obj)

    # Join building and joining wall objects.
    bpy.ops.object.select_all(action='DESELECT')
    joined_objs = [joined_buildings] + joining_wall_objects
    for obj in joined_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = joined_buildings
    bpy.ops.object.join()
    final_building_obj = bpy.context.active_object

    merge_vertices_by_distance(final_building_obj, MERGE_THRESHOLD)

    # IMPORTANT: We intentionally skip any floor cleanup so that the vertical faces of the bottom extrusion remain.
    select_bottom_extrusion_faces(final_building_obj, extrusion_height, tol=0.2)
    filter_ground_floor_faces(final_building_obj, min_abs_x=0.7)

    # --- Mark (assign materials to) the various elements ---
    assign_face_materials(final_building_obj, extrusion_height, mats)
    for wire_obj in wire_objects:
        assign_wire_material(wire_obj, mats)

    # Perform the final selection based on the parameter.
    select_final_elements(final_building_obj, wire_objects, FINAL_SELECTION)

    # At this point your scene contains:
    #  - A mesh object (final_building_obj) with faces marked as:
    #       Floor (Mat_Floor) – horizontal faces at z ≈ 0 (now violet).
    #       Ceiling (Mat_Ceiling) – horizontal faces above the floor (now orange).
    #       Ground floor walls (Mat_GroundWall) – vertical walls below roughly one floor height (red).
    #       Story walls (Mat_StoryWall) – vertical walls above the ground floor (green).
    #       Joining walls (Mat_JoiningWall) – faces with normals mainly along Y (blue).
    #  - Curve objects for the wires with the Mat_Wire material (yellow).
    # And only the element(s) specified by FINAL_SELECTION will be selected.

if __name__ == '__main__':
    main()
