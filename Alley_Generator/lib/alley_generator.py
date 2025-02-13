import bpy
import bmesh
import random
import colorsys
import math
from dataclasses import dataclass
from typing import Tuple, List

# -------------------------------------------------------------------------------
# Internal Configuration Dataclass (not exposed publicly)
# -------------------------------------------------------------------------------

@dataclass(frozen=True)
class _AlleyConfig:
    num_buildings: int
    min_stories: int
    max_stories: int
    extrusion_height: float
    building_depth: float
    min_building_width: float
    max_building_width: float
    spacing: float

    replace_storefront: bool
    storefront_text_name: str

    random_seed: int
    merge_threshold: float

    # Wire loom parameters
    wire_story_count: int
    wire_bundle_count: int
    base_sag: float
    sag_randomness: float
    base_offset: float
    offset_randomness: float
    left_loom_randomness: float
    right_loom_randomness: float

    # Floor extrusion parameters
    extrude_amount: float
    ground_floor_separation: float

# -------------------------------------------------------------------------------
# Material Setup
# -------------------------------------------------------------------------------

def _get_or_create_material(name: str, hsv: tuple, diffuse_alpha: float = 1.0) -> bpy.types.Material:
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    r, g, b = colorsys.hsv_to_rgb(*hsv)
    mat.diffuse_color = (r, g, b, diffuse_alpha)
    return mat

def _setup_materials() -> dict:
    return {
        "floor":   _get_or_create_material("Mat_Floor",      (0.75, 1, 1)),
        "ceiling": _get_or_create_material("Mat_Ceiling",    (0.08, 1, 1)),
        "ground":  _get_or_create_material("Mat_GroundWall", (0.0, 1, 1)),
        "story":   _get_or_create_material("Mat_StoryWall",  (0.33, 1, 1)),
        "joining": _get_or_create_material("Mat_JoiningWall",(0.66, 1, 1)),
        "wire":    _get_or_create_material("Mat_Wire",       (0.1667, 1, 1))
    }

# -------------------------------------------------------------------------------
# Building & Geometry Creation Functions
# -------------------------------------------------------------------------------

def _create_building(building_width: float, building_depth: float,
                     extrusion_height: float, num_extrusions: int,
                     location: Tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1, location=location)
    building_obj = bpy.context.active_object
    building_obj.scale.x = building_width
    building_obj.scale.y = building_depth
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(building_obj.data)
    half_width = building_width / 2.0
    for edge in bm.edges:
        if all(v.co.x < -0.9 * half_width for v in edge.verts):
            edge.select = True
        if all(v.co.x > 0.9 * half_width for v in edge.verts):
            edge.select = True
    bmesh.update_edit_mesh(building_obj.data)
    bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, extrusion_height)})
    for _ in range(num_extrusions - 1):
        bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, extrusion_height)})
    bpy.ops.object.mode_set(mode='OBJECT')
    return building_obj

def _create_wire_loom(building_location: Tuple[float, float, float], building_width: float,
                        extrusion_height: float, num_extrusions: int,
                        config: _AlleyConfig) -> bpy.types.Object:
    curve_data = bpy.data.curves.new(name="WireLoom", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.02
    curve_data.bevel_resolution = 4
    curve_data.resolution_u = 12

    max_story = min(config.wire_story_count, num_extrusions)
    for floor in range(1, max_story + 1):
        z = building_location[2] + floor * extrusion_height
        left_offset = random.uniform(-config.left_loom_randomness, config.left_loom_randomness)
        right_offset = random.uniform(-config.right_loom_randomness, config.right_loom_randomness)
        for _ in range(config.wire_bundle_count):
            current_sag = config.base_sag + random.uniform(-config.sag_randomness, config.sag_randomness)
            left_individual = random.uniform(-config.offset_randomness, config.offset_randomness)
            right_individual = random.uniform(-config.offset_randomness, config.offset_randomness)
            left_y = config.base_offset + left_offset + left_individual
            right_y = config.base_offset + right_offset + right_individual
            start_point = (building_location[0] - building_width / 2, building_location[1] + left_y, z)
            end_point   = (building_location[0] + building_width / 2, building_location[1] + right_y, z)
            spline = curve_data.splines.new('BEZIER')
            spline.bezier_points.add(2)
            bp0, bp1, bp2 = spline.bezier_points
            bp0.co = start_point
            bp2.co = end_point
            mid = ((start_point[0] + end_point[0]) / 2,
                   (start_point[1] + end_point[1]) / 2,
                   (start_point[2] + end_point[2]) / 2 - current_sag)
            bp1.co = mid
            for bp in spline.bezier_points:
                bp.handle_left_type = bp.handle_right_type = 'AUTO'
    loom_obj = bpy.data.objects.new("WireLoom", curve_data)
    bpy.context.collection.objects.link(loom_obj)
    return loom_obj

def _create_joining_wall(cell1: dict, cell2: dict,
                         building_depth: float, extrusion_height: float) -> List[bpy.types.Object]:
    joining_objects = []
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
    build_wall_edge(cell1['width'] / 2, cell2['width'] / 2)
    return joining_objects

# -------------------------------------------------------------------------------
# Mesh Utility Functions
# -------------------------------------------------------------------------------

def _join_objects(obj_list: List[bpy.types.Object]) -> bpy.types.Object:
    for obj in obj_list:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = obj_list[0]
    bpy.ops.object.join()
    return bpy.context.active_object

def _remove_doubles(obj: bpy.types.Object, merge_threshold: float) -> None:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

def _extrude_floor(obj: bpy.types.Object, extrude_amount: float,
                   ground_floor_separation: float, extrusion_height: float) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        if abs(face.normal.z) < 0.3:
            center_z = sum(v.co.z for v in face.verts) / len(face.verts)
            if 0.2 < center_z < (extrusion_height - 0.2):
                face.select = True
    bmesh.update_edit_mesh(mesh)
    bpy.ops.mesh.extrude_region_shrink_fatten(TRANSFORM_OT_shrink_fatten={"value": extrude_amount})
    bpy.ops.transform.resize(value=(ground_floor_separation, 1, 1), constraint_axis=(True, False, False))
    bpy.ops.object.mode_set(mode='OBJECT')

def _filter_selection_by_normal_x(obj: bpy.types.Object, threshold: float = 0.7) -> None:
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

def _assign_materials(final_obj: bpy.types.Object, mats: dict,
                      extrusion_height: float) -> None:
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

def _assign_wire_material(wire_objs: List[bpy.types.Object], mat_wire: bpy.types.Material) -> None:
    for wire_obj in wire_objs:
        curve = wire_obj.data
        curve.materials.clear()
        curve.materials.append(mat_wire)

# -------------------------------------------------------------------------------
# Storefront Replacement
# -------------------------------------------------------------------------------

def _replace_storefront_faces(final_obj: bpy.types.Object, storefront_module, initial_seed: int) -> List[bpy.types.Object]:
    area_threshold = 0.05
    bpy.context.view_layer.objects.active = final_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(final_obj.data)
    storefront_faces_data = []
    for face in bm.faces[:]:
        if face.material_index == 1 and abs(face.normal.z) < 0.1 and face.calc_area() > area_threshold:
            center = face.calc_center_median()
            max_edge_length = 0.0
            loops = face.loops
            for i in range(len(loops)):
                v1 = loops[i].vert.co
                v2 = loops[(i+1) % len(loops)].vert.co
                edge_length = (v1 - v2).length
                if edge_length > max_edge_length:
                    max_edge_length = edge_length
            plane_size = max_edge_length if max_edge_length > 0 else 2.0
            normal_xy = face.normal.copy()
            normal_xy.z = 0
            if normal_xy.length != 0:
                normal_xy.normalize()
                angle = math.degrees(math.atan2(normal_xy.y, normal_xy.x))
            else:
                angle = 0
            storefront_faces_data.append({
                'center': center.copy(),
                'plane_size': plane_size,
                'plane_orientation': angle
            })
            face.select = True
        else:
            face.select = False
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.select], context='FACES')
    bmesh.update_edit_mesh(final_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    storefront_objects = []
    # Assumes there is a "store_name_generator.py" text block in Blender
    store_name_generator = bpy.data.texts["store_name_generator.py"].as_module()

    # Start with the provided seed and increment it for each storefront.
    current_seed = initial_seed
    for data in storefront_faces_data:
        storefront_obj = storefront_module.create_empty_storefront(
            plane_size=data['plane_size'],
            plane_location=(data['center'].x, data['center'].y, data['center'].z),
            plane_orientation=data['plane_orientation'] + 90,  # adjust if needed
            sign_height=0.3,
            sign_depth=0.12,
            sign_face_depth=-0.03,
            sign_border_margin=0.012,
            pillar_width_left=0.1,
            pillar_width_right=0.1,
            pillar_depth=0.06,
            front_face_depth=-0.06,
            shutter_segments=9,
            shutter_depth=0.006,
            shutter_closed=0.2,
            shop_name=store_name_generator.generate_shop_name(seed=current_seed)
        )
        print("Storefront created successfully:", storefront_obj)
        storefront_objects.append(storefront_obj)
        current_seed += 1  # Increment the seed for the next storefront.
    return storefront_objects

# -------------------------------------------------------------------------------
# New Feature: Extract Upper Floor (Green Story Wall) Faces with Tower-Specific Tennancy
# -------------------------------------------------------------------------------
def _extract_upper_floor_faces(final_obj: bpy.types.Object, building_info: List[dict], building_depth: float) -> List[bpy.types.Object]:
    """
    Extracts the green upper floor (story wall) faces and replaces them with tennancy windows.
    Each face is assigned to a tower based on its Y coordinate, and each tower uses its
    predetermined variant (assigned during building creation) so that all floors in the tower
    share the same tennancy style.
    
    Returns:
        A list of new objects representing the tennancy windows.
    """
    area_threshold = 0.05
    bpy.context.view_layer.objects.active = final_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(final_obj.data)
    upper_faces_data = []
    for face in bm.faces[:]:
        if face.material_index == 2 and abs(face.normal.z) < 0.1 and face.calc_area() > area_threshold:
            center = face.calc_center_median()
            max_edge_length = 0.0
            for loop in face.loops:
                v1 = loop.vert.co
                v2 = loop.link_loop_next.vert.co
                edge_length = (v1 - v2).length
                if edge_length > max_edge_length:
                    max_edge_length = edge_length
            plane_size = max_edge_length if max_edge_length > 0 else 2.0
            normal_xy = face.normal.copy()
            normal_xy.z = 0
            angle = math.degrees(math.atan2(normal_xy.y, normal_xy.x)) if normal_xy.length else 0
            upper_faces_data.append({
                'center': center.copy(),
                'plane_size': plane_size,
                'plane_orientation': angle
            })
            face.select = True
        else:
            face.select = False

    # Delete the selected (upper floor) faces from the original mesh.
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.select], context='FACES')
    bmesh.update_edit_mesh(final_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Group the extracted faces by tower using the face's Y coordinate.
    towers_faces = {}  # key: tower_id, value: list of face data
    for data in upper_faces_data:
        face_center_y = data['center'].y
        assigned_tower = None
        min_dist = float('inf')
        for tower in building_info:
            dist = abs(face_center_y - tower['center_y'])
            if dist < building_depth / 2 and dist < min_dist:
                min_dist = dist
                assigned_tower = tower
        if assigned_tower is not None:
            tower_id = assigned_tower['tower_id']
            towers_faces.setdefault(tower_id, []).append(data)

    upper_face_objects = []
    # Process each tower’s faces using its assigned variant.
    for tower in building_info:
        tower_id = tower['tower_id']
        if tower_id not in towers_faces:
            continue
        variant = tower['variant']
        try:
            tennancy_module = bpy.data.texts[f"tennancy_{variant}.py"].as_module()
        except KeyError:
            print(f"Error: 'tennancy_{variant}.py' text block not found. Upper floor extraction for tower {tower_id} aborted.")
            continue
        for data in towers_faces[tower_id]:
            window_obj = tennancy_module.create_tennancy(
                plane_size=data['plane_size'],
                plane_location=(data['center'].x, data['center'].y, data['center'].z),
                plane_rotation=(90, 0, data['plane_orientation'] + 90),
                margin_left=0.4,
                margin_right=0.4,
                margin_top=0.2,
                margin_bottom=0.2,
                extrude_short=0.03,
                extrude_long=0.05
            )
            print(f"Tennancy window created successfully for tower {tower_id} (variant {variant}):", window_obj)
            upper_face_objects.append(window_obj)
    return upper_face_objects

# -------------------------------------------------------------------------------
# Public Function: generate_alley
# -------------------------------------------------------------------------------

def generate_alley(
    num_buildings: int = 10,
    min_stories: int = 3,
    max_stories: int = 8,
    extrusion_height: float = 2.0,
    building_depth: float = 2.0,
    min_building_width: float = 1.5,
    max_building_width: float = 3.0,
    spacing: float = 2.0,
    replace_storefront: bool = True,
    storefront_text_name: str = "storefront.py",
    random_seed: int = 1245,
    merge_threshold: float = 0.0001,
    # Wire loom parameters
    wire_story_count: int = 1,
    wire_bundle_count: int = 3,
    base_sag: float = 0.3,
    sag_randomness: float = 0.2,
    base_offset: float = 0.2,
    offset_randomness: float = 0.1,
    left_loom_randomness: float = 0.2,
    right_loom_randomness: float = 0.2,
    # Floor extrusion parameters
    extrude_amount: float = 0,
    ground_floor_separation: float = 1.2,
    # New parameter for extracting upper floor faces:
    extract_upper_floor: bool = False
) -> Tuple[bpy.types.Object, List[bpy.types.Object], List[bpy.types.Object]]:
    """
    Generate an alley of extruded building objects with wire looms and joining walls.
    Optionally, replace selected ground faces with storefronts (via a Blender text block module)
    and optionally extract green upper floor (story wall) faces into separate objects.
    
    For each tower, a unique tennancy variant (from 1 to 3) is assigned so that all floors
    within the tower share the same style while different towers may use different variants.
    
    Returns:
        A tuple containing:
          - The final joined building object.
          - A list of storefront objects (empty if storefront replacement is disabled).
          - A list of extracted upper floor objects (empty if extraction is disabled).
    """
    # Build the internal configuration.
    config = _AlleyConfig(
        num_buildings=num_buildings,
        min_stories=min_stories,
        max_stories=max_stories,
        extrusion_height=extrusion_height,
        building_depth=building_depth,
        min_building_width=min_building_width,
        max_building_width=max_building_width,
        spacing=spacing,
        replace_storefront=replace_storefront,
        storefront_text_name=storefront_text_name,
        random_seed=random_seed,
        merge_threshold=merge_threshold,
        wire_story_count=wire_story_count,
        wire_bundle_count=wire_bundle_count,
        base_sag=base_sag,
        sag_randomness=sag_randomness,
        base_offset=base_offset,
        offset_randomness=offset_randomness,
        left_loom_randomness=left_loom_randomness,
        right_loom_randomness=right_loom_randomness,
        extrude_amount=extrude_amount,
        ground_floor_separation=ground_floor_separation
    )
    # Seed the global random for other operations.
    random.seed(config.random_seed)
    mats = _setup_materials()

    building_objs = []
    wire_objs = []
    building_info = []
    current_y = 0.0

    # --- Modified Tenant Variant Assignment ---
    # Use a dedicated RNG for tenant variant assignment so that it is solely based on config.random_seed.
    tenant_rng = random.Random(config.random_seed)
    available_variants = list(range(1, 5))
    if config.num_buildings <= len(available_variants):
        tower_variants = tenant_rng.sample(available_variants, config.num_buildings)
    else:
        tower_variants = [tenant_rng.choice(available_variants) for _ in range(config.num_buildings)]
    # ------------------------------------------------

    # Create each building (tower) and its corresponding wire loom.
    for i in range(config.num_buildings):
        building_width = random.uniform(config.min_building_width, config.max_building_width)
        num_extrusions = random.randint(config.min_stories, config.max_stories)
        building_height = (num_extrusions + 1) * config.extrusion_height
        building_location = (0, current_y, 0)
        b_obj = _create_building(building_width, config.building_depth,
                                 config.extrusion_height, num_extrusions,
                                 building_location)
        building_objs.append(b_obj)
        building_info.append({
            'tower_id': i,
            'center_y': current_y,
            'width': building_width,
            'height': building_height,
            'extrusions': num_extrusions,
            'variant': tower_variants[i]
        })
        loom_obj = _create_wire_loom(building_location, building_width,
                                     config.extrusion_height, num_extrusions, config)
        wire_objs.append(loom_obj)
        current_y += config.spacing

    # Join all building objects.
    joined_buildings = _join_objects(building_objs)

    # Create and join joining wall meshes.
    joining_wall_objs = []
    for i in range(len(building_info) - 1):
        walls = _create_joining_wall(building_info[i], building_info[i+1],
                                     config.building_depth, config.extrusion_height)
        joining_wall_objs.extend(walls)
    objs_to_join = [joined_buildings] + joining_wall_objs
    final_building_obj = _join_objects(objs_to_join)

    # Cleanup and process the geometry.
    _remove_doubles(final_building_obj, config.merge_threshold)
    _extrude_floor(final_building_obj, config.extrude_amount,
                   config.ground_floor_separation, config.extrusion_height)
    _filter_selection_by_normal_x(final_building_obj, threshold=0.7)
    _assign_materials(final_building_obj, mats, config.extrusion_height)
    _assign_wire_material(wire_objs, mats["wire"])

    storefront_objects = []
    if config.replace_storefront:
        try:
            storefront_module = bpy.data.texts[config.storefront_text_name].as_module()
        except KeyError:
            print(f"Error: '{config.storefront_text_name}' text block not found.")
            storefront_module = None
        if storefront_module:
            # Pass the alley's random_seed as the starting seed for storefront names.
            storefront_objects = _replace_storefront_faces(final_building_obj, storefront_module, config.random_seed)

    upper_floor_objects = []
    if extract_upper_floor:
        upper_floor_objects = _extract_upper_floor_faces(final_building_obj, building_info, config.building_depth)

    return final_building_obj, storefront_objects, upper_floor_objects

# Optionally, define __all__ for clean imports.
__all__ = ["generate_alley"]

# -------------------------------------------------------------------------------
# End of Module
# -------------------------------------------------------------------------------

if __name__ == '__main__':
    # For testing purposes:
    final_obj, storefronts, upper_floor_objs = generate_alley(
        num_buildings=8,
        extrusion_height=2,
        min_stories=2,
        max_stories=5,
        min_building_width=2.0,
        max_building_width=4.0,
        spacing=2.0,    
        replace_storefront=True,
        extract_upper_floor=True,
    )
    print("Alley generated successfully:", final_obj)
    print("Storefront objects created:", storefronts)
    print("Upper floor objects created:", upper_floor_objs)
