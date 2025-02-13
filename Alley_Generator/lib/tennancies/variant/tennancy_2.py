import bpy

def create_tennancy(
    plane_size=2.0,
    plane_location=(0, 0, 0),
    plane_rotation=(90, 0, 0),
    margin_left=0.2,
    margin_right=0.2,
    margin_top=0.1,
    margin_bottom=0.1,
    extrude_short=0.015,
    extrude_long=0.025,
    frame_width_factor=0.05,
    frame_depth_factor=0.02,
    inset_thickness_factor=0.025,
    inset_depth_factor=0.0,
    pane_extrude_factor=0.01,
    merge_pairs=[(0, 3)]
):
    """
    Create a blank tennancy window and modify its face based on the provided parameters.
    
    Parameters:
        plane_size (float): Overall plane_size of the window.
        plane_location (tuple): Location of the window plane.
        plane_rotation (tuple): Rotation of the window plane (in degrees).
        margin_left (float): Left margin.
        margin_right (float): Right margin.
        margin_top (float): Top margin.
        margin_bottom (float): Bottom margin.
        extrude_short (float): Short extrusion value.
        extrude_long (float): Long extrusion value.
        frame_width_factor (float): Factor for the frame width.
        frame_depth_factor (float): Factor for the frame depth.
        inset_thickness_factor (float): Factor for the inset thickness.
        inset_depth_factor (float): Factor for the inset depth.
        pane_extrude_factor (float): Factor for pane extrusion.
        merge_pairs (list): List of merge pairs (tuples) for merging operations.
    
    Returns:
        The modified window object.
    """
    # Load the required modules from Blender's text datablocks.
    tennancy = bpy.data.texts["tennancy.py"].as_module()
    window = bpy.data.texts["basic_window.py"].as_module()

    # Create the blank tennancy window.
    window_obj = tennancy.create_blank_tennancy(
        plane_size=plane_size,
        plane_location=plane_location,
        plane_rotation=plane_rotation,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
        extrude_short=extrude_short,
        extrude_long=extrude_long
    )
    print("Created window object:", window_obj.name)

    # Modify the window face.
    modified_obj = window.modify_window_face(
        window_obj,
        window_size=plane_size,
        frame_width_factor=frame_width_factor,
        frame_depth_factor=frame_depth_factor,
        inset_thickness_factor=inset_thickness_factor,
        inset_depth_factor=inset_depth_factor,
        pane_extrude_factor=pane_extrude_factor,
        merge_pairs=merge_pairs
    )
    print("Modified window object:", modified_obj.name)
    
    return modified_obj

