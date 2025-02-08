"""
Storefront Creation Module
--------------------------
This module provides the function `create_empty_storefront` that creates a
storefront geometry in Blender. It is configurable via the following parameters:

    # Plane Setup
    plane_size: float = 1.0
    plane_location: Tuple[float, float, float] = (0, 0, 0)
    plane_orientation: float = 90  # in degrees

    # Base Dimensions (original values)
    # Note: **sign_height** now determines the full height of the sign face.
    # For a plane of size 1 (extending from -0.5 to +0.5 in Y), the sign
    # face will extend from Y = 0.5 (top edge) down to Y = (0.5 - sign_height).
    sign_height: float = 0.25
    sign_depth: float = 0.1
    sign_face_depth: float = -0.025
    sign_border_margin: float = 0.01

    # Pillar widths are now interpreted as the actual pillar widths.
    pillar_width_left: float = 0.4
    pillar_width_right: float = 0.4
    pillar_depth: float = 0.05

    front_face_depth: float = -0.05

    shutter_segments: int = 11
    shutter_depth: float = 0.005
    shutter_closed: float = 0.15

Usage Example (from another Blender script):
    from storefront import create_empty_storefront

    my_storefront = create_empty_storefront(
        plane_size=2.0,
        plane_location=(1, 2, 0),
        plane_orientation=45,
        sign_height=0.3,
        sign_depth=0.12,
        sign_face_depth=-0.03,
        sign_border_margin=0.012,
        pillar_width_left=0.5,
        pillar_width_right=0.5,
        pillar_depth=0.06,
        front_face_depth=-0.06,
        shutter_segments=13,
        shutter_depth=0.006,
        shutter_closed=0.18
    )
"""

import bpy
import bmesh
import math
from mathutils import Vector
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict

# ==============================================================================
# CONFIGURATION DATA CLASS
# ==============================================================================

@dataclass(frozen=True)
class Config:
    # Plane Setup
    plane_size: float = 1.0
    plane_location: Tuple[float, float, float] = (0, 0, 0)
    plane_orientation: float = 90  # degrees

    # Base Dimensions (original values)
    # Note: sign_height now determines the full height of the sign face.
    sign_height: float = 0.25
    sign_depth: float = 0.1
    sign_face_depth: float = -0.025
    sign_border_margin: float = 0.01

    # Pillar widths are now interpreted as the **width** of the pillars.
    pillar_width_left: float = 0.4
    pillar_width_right: float = 0.4
    pillar_depth: float = 0.05

    front_face_depth: float = -0.05

    shutter_segments: int = 11
    shutter_depth: float = 0.005
    shutter_closed: float = 0.15

    # Internal/Factory Parameters (constants that scale)
    shutter_extrude_distance: float = -1.6  # later scaled
    side_extrude_translation: Tuple[float, float, float] = (0, -0.1, 0)
    shutter_bisect_tolerance: float = 0.001
    shutter_edge_tolerance: float = 0.01

    # Computed fields (initialized in __post_init__)
    shutter_num_cuts: int = field(init=False)
    scale: float = field(init=False)
    # ---
    # Sign face: for a centered plane (with extents ±plane_size/2), the sign face occupies
    # the region from Y = (plane_size/2 - sign_height*scale) up to Y = plane_size/2.
    scaled_horizontal_loop_y: float = field(init=False)
    # Pillar inner edges are now computed based on the pillar width.
    scaled_vertical_loop_x_left: float = field(init=False)
    scaled_vertical_loop_x_right: float = field(init=False)
    scaled_sign_depth: float = field(init=False)
    scaled_sign_face_depth: float = field(init=False)
    scaled_sign_border_margin: float = field(init=False)
    scaled_pillar_depth: float = field(init=False)
    scaled_front_face_depth: float = field(init=False)
    scaled_shutter_depth: float = field(init=False)
    scaled_shutter_closed: float = field(init=False)
    # We now use the bottom edge of the sign face as the reference for the front sign face.
    front_face_min_y: float = field(init=False)
    top_bar_assign_threshold: float = field(init=False)
    plane_rotation: Tuple[float, float, float] = field(init=False)
    scaled_shutter_extrude_distance: float = field(init=False)

    def __post_init__(self):
        # All dimensions scale with the plane size.
        object.__setattr__(self, 'scale', self.plane_size / 1.0)
        # For a centered plane, the top edge is at +plane_size/2.
        # The sign face should be exactly 'sign_height*scale' tall.
        # Hence, the lower boundary (edge loop) is at:
        #    Y = (plane_size/2) - (sign_height*scale)
        object.__setattr__(self, 'scaled_horizontal_loop_y', (self.plane_size / 2) - (self.sign_height * self.scale))
        # -------------------------------------------------------------------
        # REWORKED PILLAR EDGE LOOP POSITIONS:
        # Previously, pillar_width_left/right were used directly as the x-coordinate.
        # Now they represent the width of the pillars. Therefore, the inner edge of the
        # left pillar is at: left edge of the plane (-plane_size/2) plus pillar_width_left.
        # Similarly, for the right pillar.
        object.__setattr__(self, 'scaled_vertical_loop_x_left', -(self.plane_size / 2) + (self.pillar_width_left * self.scale))
        object.__setattr__(self, 'scaled_vertical_loop_x_right', (self.plane_size / 2) - (self.pillar_width_right * self.scale))
        # -------------------------------------------------------------------
        object.__setattr__(self, 'scaled_sign_depth', self.sign_depth * self.scale)
        object.__setattr__(self, 'scaled_sign_face_depth', self.sign_face_depth * self.scale)
        object.__setattr__(self, 'scaled_sign_border_margin', self.sign_border_margin * self.scale)
        object.__setattr__(self, 'scaled_pillar_depth', self.pillar_depth * self.scale)
        object.__setattr__(self, 'scaled_front_face_depth', self.front_face_depth * self.scale)
        object.__setattr__(self, 'scaled_shutter_depth', self.shutter_depth * self.scale)
        object.__setattr__(self, 'scaled_shutter_closed', self.shutter_closed)
        object.__setattr__(self, 'shutter_num_cuts', (self.shutter_segments * 2) - 1)
        # Instead of a threshold computed from a pillar edge position, we now use the actual
        # pillar inner edge positions directly in our face assignment logic.
        object.__setattr__(self, 'front_face_min_y', self.scaled_horizontal_loop_y)
        object.__setattr__(self, 'top_bar_assign_threshold', self.scaled_horizontal_loop_y)
        object.__setattr__(self, 'plane_rotation', (math.radians(90), 0, math.radians(self.plane_orientation)))
        object.__setattr__(self, 'scaled_shutter_extrude_distance', self.shutter_extrude_distance * self.scale)

# ==============================================================================
# HELPER FUNCTIONS & CONTEXT MANAGERS
# ==============================================================================

@contextmanager
def edit_mode(obj: bpy.types.Object):
    """
    Context manager to safely switch into edit mode for an object and return to its previous mode.
    """
    prev_mode = obj.mode
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        yield
    finally:
        bpy.ops.object.mode_set(mode=prev_mode)

def update_bmesh(obj: bpy.types.Object) -> bmesh.types.BMesh:
    """
    Return an updated BMesh for the given object.
    """
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.update_edit_mesh(obj.data)
    return bm

def get_or_create_material(mat_name: str, color: Tuple[float, float, float, float]) -> bpy.types.Material:
    """
    Retrieve an existing material by name or create a new one with the given color.
    """
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

def compute_cut_positions(shutter_faces: List[bmesh.types.BMFace],
                          num_cuts: int,
                          tolerance: float) -> List[float]:
    """
    Compute and return the Y positions for cuts given shutter faces.
    """
    shutter_verts: Set[bmesh.types.BMVert] = {v for face in shutter_faces for v in face.verts}
    y_coords = [v.co.y for v in shutter_verts]
    if not y_coords:
        raise ValueError("No vertices found in shutter faces!")
    min_y, max_y = min(y_coords), max(y_coords)
    return [min_y + (i + 1) * (max_y - min_y) / (num_cuts + 1) for i in range(num_cuts)]

def set_material_indices_by_position(bm: bmesh.types.BMesh, config: Config) -> None:
    """
    Assign material indices to faces based on their location.
    
    Faces whose centers are above the bottom of the sign face get the sign border material.
    Faces with centers outside the region between the pillar inner edges (i.e. to the far left or right)
    get the column material, while the remaining faces are considered the front.
    """
    for face in bm.faces:
        center = face.calc_center_median()
        if center.y >= config.top_bar_assign_threshold:
            face.material_index = 0  # Sign border (to later receive the inset sign face)
        elif center.x < config.scaled_vertical_loop_x_left or center.x > config.scaled_vertical_loop_x_right:
            face.material_index = 1  # Sides: Mat_Column
        else:
            face.material_index = 2  # Front: Mat_Front

def select_faces_by_material(bm: bmesh.types.BMesh, material_index: int) -> None:
    """
    Deselect all faces and select only those with the given material index.
    """
    for face in bm.faces:
        face.select = (face.material_index == material_index)

def assign_material_to_selected_faces(bm: bmesh.types.BMesh, material_index: int) -> None:
    """
    Assign the given material index to all currently selected faces.
    """
    for face in bm.faces:
        if face.select:
            face.material_index = material_index

# ==============================================================================
# MAIN GEOMETRY CREATION FUNCTIONS
# ==============================================================================

def create_base_plane(config: Config) -> bpy.types.Object:
    """
    Create the base plane with the specified settings.
    """
    bpy.ops.mesh.primitive_plane_add(
        size=config.plane_size,
        enter_editmode=False,
        align='WORLD',
        location=config.plane_location,
        rotation=config.plane_rotation
    )
    return bpy.context.active_object

def bisect_plane_geometry(obj: bpy.types.Object, config: Config) -> None:
    """
    Bisect the plane to form loops based on the configuration settings.
    """
    with edit_mode(obj):
        bm = update_bmesh(obj)
        geom_all = list(bm.verts) + list(bm.edges) + list(bm.faces)
        # Bisect horizontally along Y = scaled_horizontal_loop_y.
        # This cut now defines the bottom edge of the sign face.
        bmesh.ops.bisect_plane(
            bm,
            geom=geom_all,
            plane_co=(0, config.scaled_horizontal_loop_y, 0),
            plane_no=(0, 1, 0),
            dist=config.shutter_bisect_tolerance * config.scale,
            use_snap_center=False,
            clear_inner=False,
            clear_outer=False
        )
        # Bisect vertically at the left and right pillar inner edges.
        for x_val in (config.scaled_vertical_loop_x_left, config.scaled_vertical_loop_x_right):
            geom_all = list(bm.verts) + list(bm.edges) + list(bm.faces)
            plane_no = (1, 0, 0) if x_val < 0 else (-1, 0, 0)
            bmesh.ops.bisect_plane(
                bm,
                geom=geom_all,
                plane_co=(x_val, 0, 0),
                plane_no=plane_no,
                dist=config.shutter_bisect_tolerance * config.scale,
                use_snap_center=False,
                clear_inner=False,
                clear_outer=False
            )
        bmesh.update_edit_mesh(obj.data)

def create_and_assign_materials(obj: bpy.types.Object, config: Config) -> None:
    """
    Create or get the required materials and assign them to the object's material slots.
    """
    # Material Colors (hard-coded for now)
    COLOR_MAT_SIGN       = (1.0, 0.2, 0.2, 1)
    COLOR_MAT_COLUMN     = (0.2, 1.0, 0.2, 1)
    COLOR_MAT_FRONT      = (0.2, 0.2, 1.0, 1)
    COLOR_MAT_SIGN_FACE  = (1.0, 0.4, 0.4, 1)
    COLOR_MAT_FRONT_FACE = (0.4, 0.4, 1.0, 1)
    COLOR_SHUTTER        = (0.8, 0.8, 0.2, 1)

    mat_sign       = get_or_create_material("Mat_Sign", COLOR_MAT_SIGN)
    mat_column     = get_or_create_material("Mat_Column", COLOR_MAT_COLUMN)
    mat_front      = get_or_create_material("Mat_Front", COLOR_MAT_FRONT)
    mat_sign_face  = get_or_create_material("Mat_Sign_face", COLOR_MAT_SIGN_FACE)
    mat_front_face = get_or_create_material("Mat_Front_face", COLOR_MAT_FRONT_FACE)
    mat_shutter    = get_or_create_material("Shutter", COLOR_SHUTTER)

    mesh = obj.data
    mesh.materials.clear()
    mesh.materials.append(mat_sign)       # index 0
    mesh.materials.append(mat_column)     # index 1
    mesh.materials.append(mat_front)      # index 2
    mesh.materials.append(mat_sign_face)  # index 3
    mesh.materials.append(mat_front_face) # index 4
    mesh.materials.append(mat_shutter)    # index 5

def cleanup_extraneous_faces(obj: bpy.types.Object, config: Config) -> None:
    """
    Remove extra faces that lie entirely above the top of the sign face region.
    """
    with edit_mode(obj):
        bm = update_bmesh(obj)

        # Remove faces fully above the top of the plane (Y > plane_size/2).
        for face in bm.faces:
            if face.calc_center_median().y > (config.plane_size / 2):
                face.select = True

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.dissolve_faces()
        bmesh.update_edit_mesh(obj.data)

        # Dissolve any small remaining faces whose centers lie above the sign face region.
        for face in bm.faces:
            if face.calc_center_median().y > config.scaled_horizontal_loop_y:
                face.select = True

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.dissolve_faces()
        bmesh.update_edit_mesh(obj.data)

def assign_face_materials(obj: bpy.types.Object, config: Config) -> None:
    """
    Assign materials to faces based on their geometric location.
    """
    with edit_mode(obj):
        bm = update_bmesh(obj)
        set_material_indices_by_position(bm, config)
        bmesh.update_edit_mesh(obj.data)

def extrude_top_bar(obj: bpy.types.Object, config: Config) -> None:
    """
    Extrude the sign border (top bar) region.
    """
    with edit_mode(obj):
        bm = update_bmesh(obj)
        select_faces_by_material(bm, material_index=0)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.extrude_faces_move()
        bpy.ops.transform.shrink_fatten(value=config.scaled_sign_depth, use_even_offset=True)
        bmesh.update_edit_mesh(obj.data)

def extrude_side_columns(obj: bpy.types.Object, config: Config) -> None:
    """
    Extrude the side column faces (Mat_Column).
    """
    mesh = obj.data
    column_mat_index = None
    for i, mat in enumerate(mesh.materials):
        if mat.name == "Mat_Column":
            column_mat_index = i
            break
    if column_mat_index is None:
        raise ValueError("Material 'Mat_Column' not found on the object!")
    with edit_mode(obj):
        bm = update_bmesh(obj)
        bpy.ops.mesh.select_all(action='DESELECT')
        select_faces_by_material(bm, material_index=column_mat_index)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.extrude_faces_move()
        bpy.ops.transform.shrink_fatten(value=config.scaled_pillar_depth, use_even_offset=True)
        bmesh.update_edit_mesh(obj.data)

def inset_and_extrude_front_sign_face(obj: bpy.types.Object, config: Config) -> None:
    """
    Inset and extrude the front sign face, then assign it the Mat_Sign_face material.
    """
    with edit_mode(obj):
        bm = update_bmesh(obj)
        bpy.ops.mesh.select_all(action='DESELECT')
        # Identify candidate faces for the sign face area.
        # Now we look for faces within the central X region (between the pillar inner edges)
        # and whose center is above the bottom boundary of the sign face.
        candidates = [
            face for face in bm.faces
            if (config.scaled_vertical_loop_x_left <= face.calc_center_median().x <= config.scaled_vertical_loop_x_right and
                face.calc_center_median().y > config.front_face_min_y and
                abs(face.normal.y) < 0.3)
        ]
        if not candidates:
            raise RuntimeError("No candidate front sign faces found!")
        front_face = min(candidates, key=lambda f: f.calc_center_median().y)
        front_face.select = True
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.inset(thickness=config.scaled_sign_border_margin)
        bpy.ops.mesh.extrude_faces_move()
        bpy.ops.transform.shrink_fatten(value=config.scaled_sign_face_depth, use_even_offset=True)
        bm = update_bmesh(obj)
        assign_material_to_selected_faces(bm, material_index=3)
        bmesh.update_edit_mesh(obj.data)

def extrude_front_faces(obj: bpy.types.Object, config: Config) -> None:
    """
    Extrude the front faces (Mat_Front) and then reassign them to Mat_Front_face.
    """
    with edit_mode(obj):
        bm = update_bmesh(obj)
        bpy.ops.mesh.select_all(action='DESELECT')
        select_faces_by_material(bm, material_index=2)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.extrude_faces_indiv()
        bpy.ops.transform.shrink_fatten(value=config.scaled_front_face_depth, use_even_offset=True)
        bm = update_bmesh(obj)
        assign_material_to_selected_faces(bm, material_index=4)
        bmesh.update_edit_mesh(obj.data)

def create_interface_vertex_group(obj: bpy.types.Object) -> None:
    """
    Create the "RedBlue_Interface" vertex group containing vertices shared by Mat_Sign and Mat_Front.
    """
    mesh = obj.data
    v2mats: Dict[int, Set[int]] = {v.index: set() for v in mesh.vertices}
    for poly in mesh.polygons:
        for vid in poly.vertices:
            v2mats[vid].add(poly.material_index)
    interface_verts = [vid for vid, mats in v2mats.items() if 0 in mats and 2 in mats]
    vg = obj.vertex_groups.get("RedBlue_Interface")
    if vg is None:
        vg = obj.vertex_groups.new(name="RedBlue_Interface")
    vg.add(interface_verts, 1.0, 'ADD')
    print(f"Created vertex group 'RedBlue_Interface' with {len(interface_verts)} vertices.")

def extrude_shutter_region(obj: bpy.types.Object, config: Config) -> None:
    """
    Extrude the shutter region based on the interface vertex group.
    """
    mesh = obj.data
    vg = obj.vertex_groups.get("RedBlue_Interface")
    if vg is None:
        raise RuntimeError("RedBlue_Interface vertex group not found!")
    with edit_mode(obj):
        bm = update_bmesh(obj)
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.select_all(action='DESELECT')
        interface_indices = [
            v.index for v in mesh.vertices
            if any(g.group == vg.index and g.weight > 0 for g in v.groups)
        ]
        for edge in bm.edges:
            if (edge.verts[0].index in interface_indices and edge.verts[1].index in interface_indices):
                if abs(edge.verts[0].co.z - edge.verts[1].co.z) < config.shutter_bisect_tolerance * config.scale:
                    edge.select = True
        bmesh.update_edit_mesh(mesh)
        bm = update_bmesh(obj)
        old_face_indices = {f.index for f in bm.faces}
        shutter_distance = abs(config.scaled_shutter_extrude_distance) * (config.scaled_shutter_closed / 2)
        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0, 0, -shutter_distance)})
        bmesh.update_edit_mesh(mesh)
        bm = update_bmesh(obj)
        new_faces = [f for f in bm.faces if f.index not in old_face_indices]
        for face in new_faces:
            face.material_index = 5  # Shutter material
        bmesh.update_edit_mesh(mesh)
        bpy.ops.mesh.select_mode(type="FACE")

def create_horizontal_cuts_and_offset_shutter(obj: bpy.types.Object, config: Config) -> None:
    """
    Create horizontal cuts in the shutter region and offset alternate edges.
    """
    mesh = obj.data
    with edit_mode(obj):
        bm = update_bmesh(obj)
        # Identify the shutter material index.
        shutter_mat_index = None
        for i, mat in enumerate(mesh.materials):
            if mat.name == "Shutter":
                shutter_mat_index = i
                break
        if shutter_mat_index is None:
            raise ValueError("Shutter material not found on the object!")
        shutter_faces = [face for face in bm.faces if face.material_index == shutter_mat_index]
        if not shutter_faces:
            raise ValueError("No shutter faces found!")
        cut_positions = compute_cut_positions(shutter_faces, config.shutter_num_cuts, config.shutter_bisect_tolerance)
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
                dist=config.shutter_bisect_tolerance * config.scale,
                use_snap_center=False,
                clear_inner=False,
                clear_outer=False
            )
        bmesh.update_edit_mesh(mesh)
        shutter_faces = [face for face in bm.faces if face.material_index == shutter_mat_index]
        cut_positions = compute_cut_positions(shutter_faces, config.shutter_num_cuts, config.shutter_bisect_tolerance)
        cut_edges: Dict[int, bmesh.types.BMEdge] = {}
        for edge in bm.edges:
            if not any(face.material_index == shutter_mat_index for face in edge.link_faces):
                continue
            if abs(edge.verts[0].co.y - edge.verts[1].co.y) < config.shutter_edge_tolerance:
                median_y = (edge.verts[0].co.y + edge.verts[1].co.y) / 2.0
                for idx, cp in enumerate(cut_positions):
                    if abs(median_y - cp) < config.shutter_edge_tolerance:
                        if idx not in cut_edges:
                            cut_edges[idx] = edge
                        break
        for idx, edge in cut_edges.items():
            edge.select = (idx % 2 == 0)
        bmesh.update_edit_mesh(mesh)
        selected_verts = {v for edge in bm.edges if edge.select for v in edge.verts}
        for v in selected_verts:
            v.co.z += config.scaled_shutter_depth
        bmesh.update_edit_mesh(mesh)

# ==============================================================================
# PUBLIC FUNCTION
# ==============================================================================

def create_empty_storefront(
    plane_size: float = 1.0,
    plane_location: Tuple[float, float, float] = (0, 0, 0),
    plane_orientation: float = 90,
    sign_height: float = 0.25,
    sign_depth: float = 0.1,
    sign_face_depth: float = -0.025,
    sign_border_margin: float = 0.01,
    pillar_width_left: float = 0.1,
    pillar_width_right: float = 0.1,
    pillar_depth: float = 0.05,
    front_face_depth: float = -0.05,
    shutter_segments: int = 13,
    shutter_depth: float = 0.005,
    shutter_closed: float = 0.2
) -> bpy.types.Object:
    """
    Create an empty storefront with the specified parameters.
    
    Returns:
        The Blender object representing the storefront.
    """
    # Initialize configuration from parameters.
    config = Config(
        plane_size=plane_size,
        plane_location=plane_location,
        plane_orientation=plane_orientation,
        sign_height=sign_height,
        sign_depth=sign_depth,
        sign_face_depth=sign_face_depth,
        sign_border_margin=sign_border_margin,
        pillar_width_left=pillar_width_left,
        pillar_width_right=pillar_width_right,
        pillar_depth=pillar_depth,
        front_face_depth=front_face_depth,
        shutter_segments=shutter_segments,
        shutter_depth=shutter_depth,
        shutter_closed=shutter_closed
    )
    
    # Create and process the geometry.
    obj = create_base_plane(config)
    bisect_plane_geometry(obj, config)
    create_and_assign_materials(obj, config)
    cleanup_extraneous_faces(obj, config)
    assign_face_materials(obj, config)
    extrude_top_bar(obj, config)
    extrude_side_columns(obj, config)
    inset_and_extrude_front_sign_face(obj, config)
    extrude_front_faces(obj, config)
    create_interface_vertex_group(obj)
    extrude_shutter_region(obj, config)
    create_horizontal_cuts_and_offset_shutter(obj, config)
    
    print("Storefront creation completed successfully.")
    return obj

# Optionally, allow testing the module directly:
if __name__ == '__main__':
    create_empty_storefront()
