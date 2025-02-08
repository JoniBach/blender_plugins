import bpy

# Convert the "storefront" text block into a module.
storefront = bpy.data.texts["storefront.py"].as_module()

# Now call the function:
storefront_obj = storefront.create_empty_storefront(
    plane_size=2.0,
    plane_location=(0, 0, 0),
    plane_orientation=45,
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
    shutter_closed=0.2
)

print("Storefront created successfully:", storefront_obj)
