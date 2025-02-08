"""
Production‐Ready Blender Building Generator

This script creates a series of extruded building objects, adds “wire looms” as curves,
generates joining walls between adjacent buildings, cleans up geometry, and finally assigns
materials based on face orientation and position.

Coding principles applied:
  - Small, focused functions
  - DRY – repeated logic has been factored out
  - Clear naming and inline documentation
  - Minimal use of bpy.ops calls wrapped inside helper functions

Before running, ensure your Blender scene is ready and that you have saved any work.
"""

import bpy
import bmesh
import random
import colorsys

# =============================================================================
# Global Constants and Parameters
# =============================================================================
MERGE_THRESHOLD = 0.0001
RANDOM_SEED = 1245
NUM_BUILDINGS = 10
MIN_STORIES = 3
MAX_STORIES = 8
EXTRUSION_HEIGHT = 2.0
MIN_BUILDING_WIDTH = 1.5
MAX_BUILDING_WIDTH = 3.0
BUILDING_DEPTH = 2.0

WIRE_STORY_COUNT = 1
WIRE_BUNDLE_COUNT = 3
BASE_SAG = 0.3
SAG_RANDOMNESS = 0.2
BASE_OFFSET = 0.5
OFFSET_RANDOMNESS = 0.1
LEFT_LOOM_RANDOMNESS = 0.2
RIGHT_LOOM_RANDOMNESS = 0.2

EXTRUDE_AMOUNT = 0
GROUND_FLOOR_SEPERATION = 1.2
SPACING = BUILDING_DEPTH

# Seed the RNG for reproducibility
random.seed(RANDOM_SEED)

# =============================================================================
# Material Setup
# =============================================================================
def get_or_create_material(name: str, hsv: tuple, diffuse_alpha: float = 1.0) -> bpy.types.Material:
    """
    Returns an existing material or creates a new one with the specified HSV color.
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    r, g, b = colorsys.hsv_to_rgb(*hsv)
    mat.diffuse_color = (r, g, b, diffuse_alpha)
    return mat

def setup_materials() -> dict:
    """
    Sets up and returns a dictionary of materials.
    Material indices (in the final mesh) will be:
      0: floor, 1: ground, 2: story, 3: joining, 4: ceiling.
    """
    return {
        "floor":   get_or_create_material("Mat_Floor",      (0.75, 1, 1)),
        "ceiling": get_or_create_material("Mat_Ceiling",    (0.08, 1, 1)),
        "ground":  get_or_create_material("Mat_GroundWall", (0.0, 1, 1)),
        "story":   get_or_create_material("Mat_StoryWall",  (0.33, 1, 1)),
        "joining": get_or_create_material("Mat_JoiningWall",(0.66, 1, 1)),
        "wire":    get_or_create_material("Mat_Wire",       (0.1667, 1, 1))
    }

# =============================================================================
# Building Creation Functions
# =============================================================================
def create_building(building_width: float, building_depth: float,
                    extrusion_height: float, num_extrusions: int,
                    location: tuple) -> bpy.types.Object:
    """
    Creates a building as a plane that is scaled and then extruded on its left/right edges.
    """
    bpy.ops.mesh.primitive_plane_add(size=1, location=location)
    building_obj = bpy.context.active_object
    building_obj.scale.x = building_width
    building_obj.scale.y = building_depth
    bpy.ops.object.transform_apply(scale=True)

    # Enter edit mode to select edges for extrusion
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(building_obj.data)
    half_width = building_width / 2.0
    for edge in bm.edges:
        # Select edges on the far left and right based on their x-coordinates
        if all(v.co.x < -0.9 * half_width for v in edge.verts):
            edge.select = True
        if all(v.co.x > 0.9 * half_width for v in edge.verts):
            edge.select = True
    bmesh.update_edit_mesh(building_obj.data)

    # Extrude the selected edges upward to build the stories.
    bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, extrusion_height)})
    for _ in range(num_extrusions - 1):
        bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, extrusion_height)})
    bpy.ops.object.mode_set(mode='OBJECT')
    return building_obj

def create_wire_loom(building_location: tuple, building_width: float,
                     extrusion_height: float, num_extrusions: int) -> bpy.types.Object:
    """
    Creates a “wire loom” curve object along the building for a limited number of stories.
    """
    curve_data = bpy.data.curves.new(name="WireLoom", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.02
    curve_data.bevel_resolution = 4
    curve_data.resolution_u = 12

    max_story = min(WIRE_STORY_COUNT, num_extrusions)
    for floor in range(1, max_story + 1):
        z = building_location[2] + floor * extrusion_height
        left_loom_offset = random.uniform(-LEFT_LOOM_RANDOMNESS, LEFT_LOOM_RANDOMNESS)
        right_loom_offset = random.uniform(-RIGHT_LOOM_RANDOMNESS, RIGHT_LOOM_RANDOMNESS)
        for _ in range(WIRE_BUNDLE_COUNT):
            current_sag = BASE_SAG + random.uniform(-SAG_RANDOMNESS, SAG_RANDOMNESS)
            left_individual = random.uniform(-OFFSET_RANDOMNESS, OFFSET_RANDOMNESS)
            right_individual = random.uniform(-OFFSET_RANDOMNESS, OFFSET_RANDOMNESS)
            left_y = BASE_OFFSET + left_loom_offset + left_individual
            right_y = BASE_OFFSET + right_loom_offset + right_individual
            start_point = (building_location[0] - building_width / 2, building_location[1] + left_y, z)
            end_point   = (building_location[0] + building_width / 2, building_location[1] + right_y, z)
            # Create a Bezier spline with three points (start, mid, end)
            spline = curve_data.splines.new('BEZIER')
            spline.bezier_points.add(2)  # creates 3 points in total
            bp0, bp1, bp2 = spline.bezier_points
            bp0.co = start_point
            bp2.co = end_point
            mid = ( (start_point[0] + end_point[0]) / 2,
                    (start_point[1] + end_point[1]) / 2,
                    (start_point[2] + end_point[2]) / 2 - current_sag )
            bp1.co = mid
            for bp in spline.bezier_points:
                bp.handle_left_type = bp.handle_right_type = 'AUTO'

    loom_obj = bpy.data.objects.new("WireLoom", curve_data)
    bpy.context.collection.objects.link(loom_obj)
    return loom_obj

def create_joining_wall(cell1: dict, cell2: dict,
                        building_depth: float, extrusion_height: float) -> list:
    """
    Creates subdivided joining wall meshes between two adjacent buildings.
    Returns a list of wall objects (one for each edge that needs a wall).
    """
    joining_objects = []
    # Choose the cell with the lower height for visible wall height and subdivisions.
    if cell1['height'] <= cell2['height']:
        junction_y = cell1['center_y'] + building_depth / 2
        visible_height = cell1['height'] - extrusion_height
        subdivisions = cell1['extrusions']
    else:
        junction_y = cell2['center_y'] - building_depth / 2
        visible_height = cell2['height'] - extrusion_height
        subdivisions = cell2['extrusions']

    def build_wall_edge(edge1: float, edge2: float):
        if abs(edge1 - edge2) > 1e-4:
            verts = []
            faces = []
            for j in range(subdivisions + 1):
                z = (visible_height * j) / subdivisions
                # Ensure the smaller coordinate is first
                verts.extend([
                    (min(edge1, edge2), junction_y, z),
                    (max(edge1, edge2), junction_y, z)
                ])
            for j in range(subdivisions):
                a = 2 * j
                b = a + 1
                c = 2 * (j + 1) + 1
                d = 2 * (j + 1)
                faces.append((a, b, c, d))
            mesh = bpy.data.meshes.new("JoiningWallSubdivMesh")
            wall_obj = bpy.data.objects.new("JoiningWallSubdiv", mesh)
            bpy.context.collection.objects.link(wall_obj)
            mesh.from_pydata(verts, [], faces)
            mesh.update()
            joining_objects.append(wall_obj)

    build_wall_edge(-cell1['width'] / 2, -cell2['width'] / 2)
    build_wall_edge( cell1['width'] / 2,  cell2['width'] / 2)
    return joining_objects

# =============================================================================
# Mesh Utility Functions
# =============================================================================
def join_objects(obj_list: list) -> bpy.types.Object:
    """
    Joins all objects in obj_list into a single object.
    The first object in the list becomes the active object.
    """
    for obj in obj_list:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = obj_list[0]
    bpy.ops.object.join()
    return bpy.context.active_object

def remove_doubles(obj: bpy.types.Object, merge_threshold: float) -> None:
    """
    Removes duplicate vertices within the merge_threshold.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

def extrude_floor(obj: bpy.types.Object, extrusion_amount: float,
                  ground_floor_seperation: float, extrusion_height: float) -> None:
    """
    Extrudes floor elements (selected based on face orientation and height) and then scales them.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    # Select faces that are nearly vertical and at an appropriate height.
    for face in bm.faces:
        if abs(face.normal.z) < 0.3:
            center_z = sum(v.co.z for v in face.verts) / len(face.verts)
            if 0.2 < center_z < (extrusion_height - 0.2):
                face.select = True
    bmesh.update_edit_mesh(mesh)
    bpy.ops.mesh.extrude_region_shrink_fatten(TRANSFORM_OT_shrink_fatten={"value": extrusion_amount})
    bpy.ops.transform.resize(value=(ground_floor_seperation, 1, 1), constraint_axis=(True, False, False))
    bpy.ops.object.mode_set(mode='OBJECT')

def filter_selection_by_normal_x(obj: bpy.types.Object, threshold: float = 0.7) -> None:
    """
    Deselects any face that is selected but has an x-component of its normal below the threshold.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        if face.select and abs(face.normal.x) < threshold:
            face.select = False
    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')

def assign_materials(final_obj: bpy.types.Object, mats: dict,
                     extrusion_height: float) -> None:
    """
    Clears the material slots of final_obj, assigns the materials in a specific order,
    and then assigns a material index to each face based on its normal and center position.
    The expected order is: floor (0), ground (1), story (2), joining (3), ceiling (4).
    """
    mesh = final_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    mesh.materials.clear()
    material_order = [mats["floor"], mats["ground"], mats["story"], mats["joining"], mats["ceiling"]]
    for mat in material_order:
        mesh.materials.append(mat)

    for face in bm.faces:
        center = face.calc_center_median()
        normal = face.normal
        if abs(normal.z) > 0.9:
            # Floor if very low, otherwise ceiling
            face.material_index = 0 if center.z < 0.1 else 4
        else:
            if abs(normal.x) > 0.7:
                face.material_index = 1 if center.z < (extrusion_height + 0.1) else 2
            elif abs(normal.y) > 0.7:
                face.material_index = 3
            else:
                face.material_index = 1
    bm.to_mesh(mesh)
    bm.free()

def assign_wire_material(wire_objs: list, mat_wire: bpy.types.Material) -> None:
    """
    Clears any existing materials on the wire (curve) objects and assigns the wire material.
    """
    for wire_obj in wire_objs:
        curve = wire_obj.data
        curve.materials.clear()
        curve.materials.append(mat_wire)

def select_faces_by_material(obj: bpy.types.Object, material_index: int) -> None:
    """
    Selects (in edit mode) all faces of the object that have the given material index.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        if face.material_index == material_index:
            face.select = True
    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')

# =============================================================================
# Main Routine
# =============================================================================
def main() -> None:
    mats = setup_materials()

    # Global parameters used in building creation.
    extrusion_height = EXTRUSION_HEIGHT
    min_width = MIN_BUILDING_WIDTH
    max_width = MAX_BUILDING_WIDTH
    building_depth = BUILDING_DEPTH
    spacing = SPACING

    building_objs = []
    wire_objs = []
    building_info = []
    current_y = 0.0

    # Create each building and its corresponding wire loom.
    for _ in range(NUM_BUILDINGS):
        building_width = random.uniform(min_width, max_width)
        num_extrusions = random.randint(MIN_STORIES, MAX_STORIES)
        building_height = (num_extrusions + 1) * extrusion_height
        building_location = (0, current_y, 0)

        building_obj = create_building(building_width, building_depth,
                                       extrusion_height, num_extrusions,
                                       building_location)
        building_objs.append(building_obj)
        building_info.append({
            'center_y': current_y,
            'width': building_width,
            'height': building_height,
            'extrusions': num_extrusions
        })

        loom_obj = create_wire_loom(building_location, building_width,
                                    extrusion_height, num_extrusions)
        wire_objs.append(loom_obj)
        current_y += spacing

    # Join all building objects into one.
    joined_buildings = join_objects(building_objs)

    # Create and join the joining wall meshes between adjacent buildings.
    joining_wall_objs = []
    for i in range(len(building_info) - 1):
        walls = create_joining_wall(building_info[i], building_info[i + 1],
                                    building_depth, extrusion_height)
        joining_wall_objs.extend(walls)

    objs_to_join = [joined_buildings] + joining_wall_objs
    final_building_obj = join_objects(objs_to_join)

    # Cleanup mesh by merging duplicate vertices.
    remove_doubles(final_building_obj, MERGE_THRESHOLD)

    # Extrude the floor faces and resize.
    extrude_floor(final_building_obj, EXTRUDE_AMOUNT, GROUND_FLOOR_SEPERATION, extrusion_height)

    # Filter selection: deselect faces that are selected but have a weak x normal.
    filter_selection_by_normal_x(final_building_obj, threshold=0.7)

    # Assign materials to faces based on orientation and position.
    assign_materials(final_building_obj, mats, extrusion_height)

    # Assign the wire material to all wire loom objects.
    assign_wire_material(wire_objs, mats["wire"])

    # As a final step, select all faces with material index 1 (e.g. ground faces)
    select_faces_by_material(final_building_obj, material_index=1)

if __name__ == "__main__":
    main()
