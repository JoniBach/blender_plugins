"""
window_module.py

An exportable module for modifying window components in Blender.
This module provides the function:
  - modify_window_face: Modify the window face (inset, subdivide, extrude) and assign materials.

Note: This module assumes that a text datablock named "tennancy.py" exists in your Blender file 
if you are using its functionality elsewhere to create the blank tennancy.
"""

import bpy
import bmesh

__all__ = ["modify_window_face"]

def modify_window_face(obj,
                       window_size=2.0,
                       frame_width_factor=0.05,    # (Reserved for future use)
                       frame_depth_factor=0.02,    # (Reserved for future use)
                       inset_thickness_factor=0.025,
                       inset_depth_factor=0.0,
                       pane_extrude_factor=0.01,
                       merge_pairs=[(2, 3)]):
    """
    Modify the window face by:
      1. Selecting the window face (material "Mat_Window_Face") and subdividing it.
      2. Optionally merging specific subdivided faces.
      3. Insetting the remaining faces individually.
      4. Extruding each inset face inward (along -face.normal) by an amount scaled to the window size.
      5. Identifying and assigning new materials to the extruded cap faces.
    
    Parameters:
        obj (Object): The Blender object representing the window.
        window_size (float): The base size of the window.
        frame_width_factor (float): Unused placeholder for frame width.
        frame_depth_factor (float): Unused placeholder for frame depth.
        inset_thickness_factor (float): Fraction of window_size for inset thickness.
        inset_depth_factor (float): Fraction of window_size for inset depth.
        pane_extrude_factor (float): Fraction of window_size for pane extrusion.
        merge_pairs (list): List of tuples indicating which face indices to merge.
    
    Returns:
        obj: The modified window object.
    """
    # Calculate actual dimensions based on the window size.
    frame_width   = frame_width_factor   * window_size  # (Currently not used)
    frame_depth   = frame_depth_factor   * window_size  # (Currently not used)
    inset_thickness = inset_thickness_factor * window_size
    inset_depth     = inset_depth_factor     * window_size
    pane_extrude    = pane_extrude_factor    * window_size

    # Ensure the object is active and switch to Edit Mode.
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    
    # Deselect all faces.
    for f in bm.faces:
        f.select = False

    # --- Step 1: Select and subdivide the window face ---
    # Look up the material index for "Mat_Window_Face".
    mat_index = None
    for i, mat in enumerate(obj.data.materials):
        if mat.name == "Mat_Window_Face":
            mat_index = i
            break
    if mat_index is None:
        raise ValueError("Material 'Mat_Window_Face' not found")
    
    # Select all faces that use the "Mat_Window_Face".
    for f in bm.faces:
        if f.material_index == mat_index:
            f.select = True

    # Subdivide the selected face(s) once.
    bpy.ops.mesh.subdivide(number_cuts=1)
    bmesh.update_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    
    # Gather subdivided faces that use the Mat_Window_Face material.
    sub_faces = [f for f in bm.faces if f.material_index == mat_index]
    print("Number of subdivided faces:", len(sub_faces))
    
    # --- Step 2: Merge designated windows (sub-faces) if requested ---
    if merge_pairs:
        merge_edges = []
        for pair in merge_pairs:
            # Check that the indices are in range.
            if max(pair) < len(sub_faces):
                face_a = sub_faces[pair[0]]
                face_b = sub_faces[pair[1]]
                common_edges = [edge for edge in face_a.edges if edge in face_b.edges]
                if common_edges:
                    merge_edges.extend(common_edges)
                    print(f"Merging faces {pair[0]} and {pair[1]} via edges: {common_edges}")
                else:
                    print("No common edge found between faces", pair[0], "and", pair[1])
            else:
                print("Merge pair", pair, "is out of range for the subdivided faces.")
        if merge_edges:
            bmesh.ops.dissolve_edges(bm, edges=merge_edges, use_verts=True)
            bmesh.update_edit_mesh(me)
    
    # After merging, get the remaining faces that still have the window face material.
    remaining_faces = [f for f in bm.faces if f.material_index == mat_index]
    print("Remaining faces after merge:", len(remaining_faces))
    
    # --- Step 3: Inset the remaining faces individually ---
    inset_result = bmesh.ops.inset_individual(
        bm,
        faces=remaining_faces,
        thickness=inset_thickness,
        depth=inset_depth,
        use_even_offset=True
    )
    bmesh.update_edit_mesh(me)
    inset_faces = inset_result.get("faces", [])
    print("Inset faces count:", len(inset_faces))
    
    # --- Step 4: Create or update the "Mat_Window_Frame" material ---
    yellow = (1.0, 1.0, 0.0, 1.0)  # RGBA yellow
    green  = (0.0, 1.0, 0.0, 1.0)  # RGBA green
    pane_mat = bpy.data.materials.get("Mat_Window_Frame")
    if pane_mat is None:
        pane_mat = bpy.data.materials.new(name="Mat_Window_Frame")
    pane_mat.use_nodes = False
    pane_mat.diffuse_color = green

    # Ensure the object uses this material.
    pane_mat_index = None
    for i, mat in enumerate(obj.data.materials):
        if mat.name == "Mat_Window_Frame":
            pane_mat_index = i
            break
    if pane_mat_index is None:
        obj.data.materials.append(pane_mat)
        pane_mat_index = len(obj.data.materials) - 1

    # --- Step 5: Extrude each inset face inward and assign the cap material ---
    cap_faces = []
    for face in inset_faces:
        # Extrude the face region.
        res = bmesh.ops.extrude_face_region(bm, geom=[face])
        extruded_geom = res["geom"]
        # Get the vertices from the extruded geometry.
        extruded_verts = [elem for elem in extruded_geom if isinstance(elem, bmesh.types.BMVert)]
        # Extrude inward along -face.normal.
        translate_vec = -face.normal * pane_extrude
        bmesh.ops.translate(bm, verts=extruded_verts, vec=translate_vec)
        bmesh.update_edit_mesh(me)
        
        # Identify the new (cap) face among the extruded geometry.
        extruded_faces = [elem for elem in extruded_geom 
                          if isinstance(elem, bmesh.types.BMFace) and elem != face]
        if not extruded_faces:
            continue
        
        # Choose the face with the greatest inward offset.
        cap_face = None
        max_offset = -1e6
        orig_center = face.calc_center_median()
        for f in extruded_faces:
            center = f.calc_center_median()
            offset = (center - orig_center).dot(-face.normal)
            if offset > max_offset:
                max_offset = offset
                cap_face = f
        if cap_face:
            cap_faces.append(cap_face)
    
    # Assign the pane material (green) only to the new cap faces.
    for f in cap_faces:
        f.material_index = pane_mat_index

    bmesh.update_edit_mesh(me)
    
    # Create (or update) a new material named "Mat_Window_Pane" set to yellow.
    mat_window_pane = bpy.data.materials.get("Mat_Window_Pane")
    if mat_window_pane is None:
        mat_window_pane = bpy.data.materials.new(name="Mat_Window_Pane")
    mat_window_pane.use_nodes = False
    mat_window_pane.diffuse_color = yellow

    # Ensure the object uses this material.
    frame_mat_index = None
    for i, mat in enumerate(obj.data.materials):
        if mat.name == "Mat_Window_Pane":
            frame_mat_index = i
            break
    if frame_mat_index is None:
        obj.data.materials.append(mat_window_pane)
        frame_mat_index = len(obj.data.materials) - 1

    # For any face still using the original window face material, assign the new frame material.
    for f in bm.faces:
        if f.material_index == mat_index:
            f.material_index = frame_mat_index
            
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj

# === Optional demonstration usage ===
if __name__ == "__main__":
    import bpy

    # Load the tennancy module from the Blender text datablocks.
    tennancy = bpy.data.texts["tennancy.py"].as_module()

    size = 2.0
    # Create the blank tennancy window.
    window_obj = tennancy.create_blank_tennancy(
        plane_size=size,
        plane_location=(0, 0, 0),
        plane_rotation=(90, 0, 0),
        margin_left=0.2,
        margin_right=0.2,
        margin_top=0.1,
        margin_bottom=0.1,
        extrude_short=0.015,
        extrude_long=0.025
    )
    print("Created window object:", window_obj.name)

    # Modify the window face.
    modified_obj = modify_window_face(
        window_obj,
        window_size=size,
        frame_width_factor=0.05,
        frame_depth_factor=0.02,
        inset_thickness_factor=0.025,
        inset_depth_factor=0.0,
        pane_extrude_factor=0.01,
        merge_pairs=[(0, 1)]  # Adjust merge pairs as necessary.
    )
    print("Modified window object:", modified_obj.name)
