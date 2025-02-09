import bpy

# # Convert the "storefront" text block into a module.
# storefront = bpy.data.texts["storefront.py"].as_module()

# # Now call the function:
# storefront_obj = storefront.create_empty_storefront(
#     plane_size=2.0,
#     plane_location=(0, 0, 0),
#     plane_orientation=45,
#     sign_height=0.3,
#     sign_depth=0.12,
#     sign_face_depth=-0.03,
#     sign_border_margin=0.012,
#     pillar_width_left=0.1,
#     pillar_width_right=0.1,
#     pillar_depth=0.06,
#     front_face_depth=-0.06,
#     shutter_segments=9,
#     shutter_depth=0.006,
#     shutter_closed=0.2
# )

# print("Storefront created successfully:", storefront_obj)
    

# Convert the "alley_generator.py" text block into a module.
alley = bpy.data.texts["alley_generator.py"].as_module()

# Now call the function with your desired parameters.
final_obj, storefronts = alley.generate_alley(
    num_buildings=8,
    extrusion_height=2,
    min_stories=2,
    max_stories=5,
    min_building_width=2.0,
    max_building_width=4.0,
    spacing=2.0,    
    replace_storefront=True,
    storefront_text_name="storefront.py",
    # The remaining parameters will use their default values.
)

print("Alley generated successfully:", final_obj)
print("Storefront objects created:", storefronts)

# # Convert the "alley_generator.py" text block into a module.
# sign = bpy.data.texts["store_sign.py"].as_module()

# # Now call the function with your desired parameters.
# signage = sign.generate_sign(
#       create_outline=False,
#         curviness=0.4,
#         remove_interior=True,
#         max_width=4.0,
#         max_height=2.0,
#         location=(0.0, 0.0, 0.0),
#         rotation=(0.0, 0.0, 0.0)
# )

# print("Sign objects created:", signage)
