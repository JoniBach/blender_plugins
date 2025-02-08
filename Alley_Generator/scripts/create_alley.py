import bpy
import bmesh
import random
import colorsys

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
FINAL_SELECTION = "floor"

random.seed(RANDOM_SEED)

if "Mat_Floor" in bpy.data.materials:
    mat_floor = bpy.data.materials["Mat_Floor"]
else:
    mat_floor = bpy.data.materials.new("Mat_Floor")
    r, g, b = colorsys.hsv_to_rgb(0.75, 1, 1)
    mat_floor.diffuse_color = (r, g, b, 1.0)

if "Mat_Ceiling" in bpy.data.materials:
    mat_ceiling = bpy.data.materials["Mat_Ceiling"]
else:
    mat_ceiling = bpy.data.materials.new("Mat_Ceiling")
    r, g, b = colorsys.hsv_to_rgb(0.08, 1, 1)
    mat_ceiling.diffuse_color = (r, g, b, 1.0)

if "Mat_GroundWall" in bpy.data.materials:
    mat_ground = bpy.data.materials["Mat_GroundWall"]
else:
    mat_ground = bpy.data.materials.new("Mat_GroundWall")
    r, g, b = colorsys.hsv_to_rgb(0.0, 1, 1)
    mat_ground.diffuse_color = (r, g, b, 1.0)

if "Mat_StoryWall" in bpy.data.materials:
    mat_story = bpy.data.materials["Mat_StoryWall"]
else:
    mat_story = bpy.data.materials.new("Mat_StoryWall")
    r, g, b = colorsys.hsv_to_rgb(0.33, 1, 1)
    mat_story.diffuse_color = (r, g, b, 1.0)

if "Mat_JoiningWall" in bpy.data.materials:
    mat_joining = bpy.data.materials["Mat_JoiningWall"]
else:
    mat_joining = bpy.data.materials.new("Mat_JoiningWall")
    r, g, b = colorsys.hsv_to_rgb(0.66, 1, 1)
    mat_joining.diffuse_color = (r, g, b, 1.0)

if "Mat_Wire" in bpy.data.materials:
    mat_wire = bpy.data.materials["Mat_Wire"]
else:
    mat_wire = bpy.data.materials.new("Mat_Wire")
    r, g, b = colorsys.hsv_to_rgb(0.1667, 1, 1)
    mat_wire.diffuse_color = (r, g, b, 1.0)

mats = {
    "floor": mat_floor,
    "ceiling": mat_ceiling,
    "ground": mat_ground,
    "story": mat_story,
    "joining": mat_joining,
    "wire": mat_wire
}

extrusion_height = EXTRUSION_HEIGHT
min_building_width = MIN_BUILDING_WIDTH
max_building_width = MAX_BUILDING_WIDTH
building_depth = BUILDING_DEPTH
spacing = SPACING

building_objects = []
wire_objects = []
building_info = []

current_y = 0.0
for i in range(NUM_BUILDINGS):
    building_width = random.uniform(min_building_width, max_buildING_WIDTH) if 'max_buildING_WIDTH' in globals() else random.uniform(min_building_width, max_building_width)
    num_extrusions = random.randint(MIN_STORIES, MAX_STORIES)
    building_height = (num_extrusions + 1) * extrusion_height
    building_location = (0, current_y, 0)
    bpy.ops.mesh.primitive_plane_add(size=1, location=building_location)
    building_obj = bpy.context.active_object
    building_obj.scale.x = building_width
    building_obj.scale.y = building_depth
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(building_obj.data)
    for edge in bm.edges:
        if all(v.co.x < -0.9 * (building_width / 2) for v in edge.verts):
            edge.select = True
        if all(v.co.x > 0.9 * (building_width / 2) for v in edge.verts):
            edge.select = True
    bmesh.update_edit_mesh(building_obj.data)
    bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, extrusion_height)})
    for _ in range(num_extrusions - 1):
        bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, extrusion_height)})
    bpy.ops.object.mode_set(mode='OBJECT')
    building_objects.append(building_obj)
    building_info.append({
        'center_y': current_y,
        'width': building_width,
        'height': building_height,
        'extrusions': num_extrusions
    })
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
        for bundle in range(WIRE_BUNDLE_COUNT):
            current_sag = BASE_SAG + random.uniform(-SAG_RANDOMNESS, SAG_RANDOMNESS)
            left_individual = random.uniform(-OFFSET_RANDOMNESS, OFFSET_RANDOMNESS)
            right_individual = random.uniform(-OFFSET_RANDOMNESS, OFFSET_RANDOMNESS)
            left_y = BASE_OFFSET + left_loom_offset + left_individual
            right_y = BASE_OFFSET + right_loom_offset + right_individual
            start_point = (building_location[0] - building_width / 2, building_location[1] + left_y, z)
            end_point = (building_location[0] + building_width / 2, building_location[1] + right_y, z)
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
    wire_objects.append(loom_obj)
    current_y += spacing

for obj in building_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = building_objects[0]
bpy.ops.object.join()
joined_buildings = bpy.context.active_object

joining_wall_objects = []
for i in range(len(building_info) - 1):
    cell1 = building_info[i]
    cell2 = building_info[i + 1]
    if cell1['height'] <= cell2['height']:
        junction_y = cell1['center_y'] + building_depth / 2
        visible_height = cell1['height'] - extrusion_height
        subdivisions = cell1['extrusions']
    else:
        junction_y = cell2['center_y'] - building_depth / 2
        visible_height = cell2['height'] - extrusion_height
        subdivisions = cell2['extrusions']
    left_edge1 = -cell1['width'] / 2
    left_edge2 = -cell2['width'] / 2
    if abs(left_edge1 - left_edge2) > 1e-4:
        verts = []
        faces = []
        for j in range(subdivisions + 1):
            z = (visible_height * j) / subdivisions
            verts.append((min(left_edge1, left_edge2), junction_y, z))
            verts.append((max(left_edge1, left_edge2), junction_y, z))
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
        joining_wall_objects.append(wall_obj)
    right_edge1 = cell1['width'] / 2
    right_edge2 = cell2['width'] / 2
    if abs(right_edge1 - right_edge2) > 1e-4:
        verts = []
        faces = []
        for j in range(subdivisions + 1):
            z = (visible_height * j) / subdivisions
            verts.append((min(right_edge1, right_edge2), junction_y, z))
            verts.append((max(right_edge1, right_edge2), junction_y, z))
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
        joining_wall_objects.append(wall_obj)

objs_to_join = [joined_buildings] + joining_wall_objects
for obj in objs_to_join:
    obj.select_set(True)
bpy.context.view_layer.objects.active = joined_buildings
bpy.ops.object.join()
final_building_obj = bpy.context.active_object

bpy.ops.object.mode_set(mode='OBJECT')
mesh = final_building_obj.data
bm = bmesh.new()
bm.from_mesh(mesh)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=MERGE_THRESHOLD)
bm.to_mesh(mesh)
bm.free()
mesh.update()

bpy.context.view_layer.objects.active = final_building_obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='FACE')
bpy.ops.mesh.select_all(action='DESELECT')
mesh = final_building_obj.data
bm = bmesh.from_edit_mesh(mesh)
for face in bm.faces:
    if abs(face.normal.z) < 0.3:
        center_z = sum(v.co.z for v in face.verts) / len(face.verts)
        if 0.2 < center_z < (extrusion_height - 0.2):
            face.select = True
bmesh.update_edit_mesh(mesh)
bpy.ops.mesh.extrude_region_shrink_fatten(TRANSFORM_OT_shrink_fatten={"value": EXTRUDE_AMOUNT})
bpy.ops.transform.resize(value=(GROUND_FLOOR_SEPERATION, 1, 1), constraint_axis=(True, False, False))
bpy.ops.object.mode_set(mode='OBJECT')

bpy.context.view_layer.objects.active = final_building_obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='FACE')
mesh = final_building_obj.data
bm = bmesh.from_edit_mesh(mesh)
for face in bm.faces:
    if face.select:
        if abs(face.normal.x) < 0.7:
            face.select = False
bmesh.update_edit_mesh(mesh)
bpy.ops.object.mode_set(mode='OBJECT')

mesh = final_building_obj.data
bm = bmesh.new()
bm.from_mesh(mesh)
mesh.materials.clear()
mesh.materials.append(mat_floor)
mesh.materials.append(mat_ground)
mesh.materials.append(mat_story)
mesh.materials.append(mat_joining)
mesh.materials.append(mat_ceiling)
for face in bm.faces:
    center = face.calc_center_median()
    normal = face.normal
    if abs(normal.z) > 0.9:
        if center.z < 0.1:
            face.material_index = 0
        else:
            face.material_index = 4
    else:
        if abs(normal.x) > 0.7:
            if center.z < (extrusion_height + 0.1):
                face.material_index = 1
            else:
                face.material_index = 2
        elif abs(normal.y) > 0.7:
            face.material_index = 3
        else:
            face.material_index = 1
bm.to_mesh(mesh)
bm.free()

for wire_obj in wire_objects:
    curve = wire_obj.data
    curve.materials.clear()
    curve.materials.append(mat_wire)

bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = final_building_obj
final_building_obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='FACE')
bpy.ops.mesh.select_all(action='DESELECT')
mesh = final_building_obj.data
bm = bmesh.from_edit_mesh(mesh)
for face in bm.faces:
    if face.material_index == 1:
        face.select = True
bmesh.update_edit_mesh(mesh)
bpy.ops.object.mode_set(mode='OBJECT')
