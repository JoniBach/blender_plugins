import bpy
import bmesh
import math
from mathutils import Vector
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple, Dict

# ============================================================================== 
# SECTION 1: CONFIGURATION DATA CLASS 
# ============================================================================== 
@dataclass(frozen=True)
class WindowConfig:
    plane_size: float = 2.0
    plane_location: Tuple[float, float, float] = (0, 0, 0)
    # Rotation is provided in degrees (converted to radians in the module)
    plane_rotation: Tuple[float, float, float] = (math.radians(90), 0, 0)
    margin_left: float = 0.5
    margin_right: float = 0.5
    margin_top: float = 0.5
    margin_bottom: float = 0.5
    extrude_short: float = 0.03
    extrude_long: float = 0.05
    # Material names
    mat_window_face: str = "Mat_Window_Face"
    mat_margin_long: str = "Mat_Window_Margin_Long"
    mat_margin_short: str = "Mat_Window_Margin_Short"
    # Material colors (RGBA)
    color_window_face: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    color_margin_long: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    color_margin_short: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

# ============================================================================== 
# SECTION 2: HELPER FUNCTIONS & CONTEXT MANAGERS 
# ============================================================================== 
@contextmanager
def edit_mode(obj: bpy.types.Object):
    """A context manager to temporarily set the object into EDIT mode."""
    prev_mode = obj.mode
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        yield
    finally:
        bpy.ops.object.mode_set(mode=prev_mode)

def update_bmesh(obj: bpy.types.Object) -> bmesh.types.BMesh:
    """Return the updated BMesh for the given object."""
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.update_edit_mesh(obj.data)
    return bm

def get_or_create_material(mat_name: str, color: Tuple[float, float, float, float]) -> bpy.types.Material:
    """
    Retrieve an existing material or create a new one with the specified base color.
    The function also sets the viewport display color so that the colors show up in Solid mode.
    """
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        # Set the base color in the Principled BSDF node.
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                node.inputs["Base Color"].default_value = color
                break
        # Set the diffuse_color used for viewport display.
        mat.diffuse_color = color
    else:
        # If the material already exists, update its color settings.
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    node.inputs["Base Color"].default_value = color
                    break
        mat.diffuse_color = color
    return mat

def extrude_faces_by_material(obj: bpy.types.Object, material_index: int, extrude_amount: float) -> None:
    """
    Extrude all faces in the object that match the specified material index.
    The extrusion is done along face normals using the shrink/fatten tool.
    """
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = update_bmesh(obj)
    for face in bm.faces:
        if face.material_index == material_index:
            face.select = True
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.extrude_faces_move()
    bpy.ops.transform.shrink_fatten(value=extrude_amount, use_even_offset=True)
    bpy.ops.object.mode_set(mode='OBJECT')

def compute_grid_params(config: WindowConfig) -> Dict[str, float]:
    """
    Compute and return grid parameters based on the provided configuration.
    These include the cut positions and boundary values.
    """
    size = config.plane_size
    # Default cut positions if margins were 1
    x_left_default = -size / 2 + size / 3
    x_right_default = size / 2 - size / 3
    y_bottom_default = -size / 2 + size / 3
    y_top_default = size / 2 - size / 3

    x_left_cut = (-size / 2) + config.margin_left * (x_left_default - (-size / 2))
    x_right_cut = (size / 2) - config.margin_right * ((size / 2) - x_right_default)
    y_bottom_cut = (-size / 2) + config.margin_bottom * (y_bottom_default - (-size / 2))
    y_top_cut = (size / 2) - config.margin_top * ((size / 2) - y_top_default)

    x_edge_left = -size / 2
    x_edge_right = size / 2
    y_edge_bottom = -size / 2
    y_edge_top = size / 2

    left_boundary = (x_edge_left + x_left_cut) / 2
    right_boundary = (x_right_cut + x_edge_right) / 2
    bottom_boundary = (y_edge_bottom + y_bottom_cut) / 2
    top_boundary = (y_top_cut + y_edge_top) / 2

    left_margin = x_left_cut - x_edge_left
    right_margin = x_edge_right - x_right_cut
    total_horizontal_margin = left_margin + right_margin

    bottom_margin = y_bottom_cut - y_edge_bottom
    top_margin = y_edge_top - y_top_cut
    total_vertical_margin = bottom_margin + top_margin

    horizontal_is_long = total_horizontal_margin >= total_vertical_margin

    return {
        "x_left_cut": x_left_cut,
        "x_right_cut": x_right_cut,
        "y_bottom_cut": y_bottom_cut,
        "y_top_cut": y_top_cut,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "bottom_boundary": bottom_boundary,
        "top_boundary": top_boundary,
        "horizontal_is_long": horizontal_is_long,
    }

def create_grid(obj: bpy.types.Object, grid_params: Dict[str, float]) -> None:
    """
    Use bisection operations to subdivide the base plane into a 3×3 grid.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Vertical bisections
    geom_all = bm.faces[:] + bm.edges[:] + bm.verts[:]
    bmesh.ops.bisect_plane(
        bm,
        geom=geom_all,
        plane_co=(grid_params["x_left_cut"], 0, 0),
        plane_no=(1, 0, 0),
        clear_inner=False,
        clear_outer=False
    )
    geom_all = bm.faces[:] + bm.edges[:] + bm.verts[:]
    bmesh.ops.bisect_plane(
        bm,
        geom=geom_all,
        plane_co=(grid_params["x_right_cut"], 0, 0),
        plane_no=(1, 0, 0),
        clear_inner=False,
        clear_outer=False
    )
    # Horizontal bisections
    geom_all = bm.faces[:] + bm.edges[:] + bm.verts[:]
    bmesh.ops.bisect_plane(
        bm,
        geom=geom_all,
        plane_co=(0, grid_params["y_bottom_cut"], 0),
        plane_no=(0, 1, 0),
        clear_inner=False,
        clear_outer=False
    )
    geom_all = bm.faces[:] + bm.edges[:] + bm.verts[:]
    bmesh.ops.bisect_plane(
        bm,
        geom=geom_all,
        plane_co=(0, grid_params["y_top_cut"], 0),
        plane_no=(0, 1, 0),
        clear_inner=False,
        clear_outer=False
    )
    bm.to_mesh(mesh)
    bm.free()

def assign_grid_face_materials(obj: bpy.types.Object,
                               grid_params: Dict[str, float],
                               config: WindowConfig,
                               idx_window_face: int,
                               idx_margin_long: int,
                               idx_margin_short: int) -> None:
    # Determine the face with the center closest to (0,0)
    min_dist = float('inf')
    center_face_index = None
    for poly in obj.data.polygons:
        cx, cy, _ = poly.center
        dist = math.hypot(cx, cy)
        if dist < min_dist:
            min_dist = dist
            center_face_index = poly.index

    # Now assign materials based on position:
    for poly in obj.data.polygons:
        if poly.index == center_face_index:
            poly.material_index = idx_window_face
        else:
            if grid_params["horizontal_is_long"]:
                if poly.center.y <= grid_params["bottom_boundary"] or poly.center.y >= grid_params["top_boundary"]:
                    poly.material_index = idx_margin_long
                elif poly.center.x <= grid_params["left_boundary"] or poly.center.x >= grid_params["right_boundary"]:
                    poly.material_index = idx_margin_short
                else:
                    poly.material_index = idx_margin_long
            else:
                if poly.center.x <= grid_params["left_boundary"] or poly.center.x >= grid_params["right_boundary"]:
                    poly.material_index = idx_margin_long
                elif poly.center.y <= grid_params["bottom_boundary"] or poly.center.y >= grid_params["top_boundary"]:
                    poly.material_index = idx_margin_short
                else:
                    poly.material_index = idx_margin_long

def create_base_plane(config: WindowConfig) -> bpy.types.Object:
    """
    Create and return a base plane using the provided configuration.
    """
    bpy.ops.mesh.primitive_plane_add(
        size=config.plane_size,
        enter_editmode=False,
        align='WORLD',
        location=config.plane_location,
        rotation=config.plane_rotation
    )
    obj = bpy.context.active_object
    obj.name = "Window_Component"
    return obj

# ============================================================================== 
# SECTION 3: MAIN GEOMETRY CREATION FUNCTION 
# ============================================================================== 
def create_blank_tennancy(
    plane_size: float = 2.0,
    plane_location: Tuple[float, float, float] = (0, 0, 0),
    plane_rotation: Tuple[float, float, float] = (90, 0, 0),  # In degrees
    margin_left: float = 0.5,
    margin_right: float = 0.5,
    margin_top: float = 0.5,
    margin_bottom: float = 0.5,
    extrude_short: float = 0.03,
    extrude_long: float = 0.05
) -> bpy.types.Object:
    """
    Create a window component consisting of a subdivided (3×3) plane with
    extruded margin faces.
    
    Unlike the previous version, this function does NOT clear the scene,
    so that new tennancy objects are simply added.
    """
    # Convert rotation from degrees to radians.
    rot_radians = (
        math.radians(plane_rotation[0]),
        math.radians(plane_rotation[1]),
        math.radians(plane_rotation[2])
    )
    config = WindowConfig(
        plane_size=plane_size,
        plane_location=plane_location,
        plane_rotation=rot_radians,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
        extrude_short=extrude_short,
        extrude_long=extrude_long
    )

    # Do not clear the scene so new tennancies are added alongside existing objects.
    # (Commented out deletion lines)
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.delete(use_global=False)

    # Create the base plane.
    obj = create_base_plane(config)

    # Set up materials.
    mat_window_face = get_or_create_material(config.mat_window_face, config.color_window_face)
    mat_margin_long = get_or_create_material(config.mat_margin_long, config.color_margin_long)
    mat_margin_short = get_or_create_material(config.mat_margin_short, config.color_margin_short)

    # Ensure the materials are added to the object's material slots.
    materials = obj.data.materials
    if config.mat_window_face not in [m.name for m in materials]:
        materials.append(mat_window_face)
    if config.mat_margin_long not in [m.name for m in materials]:
        materials.append(mat_margin_long)
    if config.mat_margin_short not in [m.name for m in materials]:
        materials.append(mat_margin_short)

    idx_window_face = obj.data.materials.find(config.mat_window_face)
    idx_margin_long = obj.data.materials.find(config.mat_margin_long)
    idx_margin_short = obj.data.materials.find(config.mat_margin_short)

    # Compute grid parameters and subdivide the plane.
    grid_params = compute_grid_params(config)
    create_grid(obj, grid_params)

    # Assign materials to each face of the grid.
    assign_grid_face_materials(obj, grid_params, config, idx_window_face, idx_margin_long, idx_margin_short)

    # Calculate effective extrusion values relative to the plane size.
    effective_extrude_short = config.plane_size * config.extrude_short
    effective_extrude_long = config.plane_size * config.extrude_long

    # Extrude the margin faces.
    extrude_faces_by_material(obj, idx_margin_long, effective_extrude_long)
    extrude_faces_by_material(obj, idx_margin_short, effective_extrude_short)

    print("Window component created successfully.")
    return obj

# ============================================================================== 
# MODULE TESTING (Run when executing the script directly) 
# ============================================================================== 
if __name__ == '__main__':
    window_obj = create_blank_tennancy(
        plane_size=2.0,
        plane_location=(0, 0, 0),
        plane_rotation=(90, 0, 0),
        margin_left=0.4,
        margin_right=0.4,
        margin_top=0.2,
        margin_bottom=0.2,
        extrude_short=0.03,
        extrude_long=0.05
    )
    print("Created window object:", window_obj.name)
