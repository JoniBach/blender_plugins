"""
Neon Shop Name Generator Library
----------------------------------

This module provides a production-ready library for generating neon shop name
objects in Blender, featuring clean code, adherence to DRY principles, and pure
functions where possible.

Usage Example:
    # Convert the "store_sign.py" text block into a module.
    sign = bpy.data.texts["store_sign.py"].as_module()

    # Now call the function with your desired parameters.
    signage = sign.generate_sign(
        create_outline=False,
        curviness=0.4,
        remove_interior=True,
        create_backing=True,
        backing_margin=0.1,
        backing_bevel_depth=0.01,
        max_width=4.0,   # Maximum width for the generated neon text.
        max_height=2.0,  # Maximum height for the generated neon text.
        location=(1.0, 2.0, 0.0),
        rotation=(0.0, 0.0, 45.0)  # 45 degrees rotation around Z
    )

    print("Sign objects created:", signage)
"""

import bpy
import random
import math  # For converting degrees to radians.
from mathutils import Vector
from typing import List, Tuple, Optional, Dict

# -------------------------------------------------------------------
# Constants: Lists of descriptive words and shop types.
# -------------------------------------------------------------------
FIRST_WORDS = [
    "Flaming", "Cozy", "Golden", "Shiny", "Majestic", "Rustic", "Elegant", "Modern",
    "Happy", "Ancient", "Royal", "Bright", "Magic", "Dreamy", "Classic", "Urban",
    "Timeless", "Charming", "Bold", "Vibrant", "Lush", "Peaceful", "Witty", "Hidden",
    "Velvet", "Silent", "Playful", "Enchanted", "Serene", "Warm", "Gleaming", "Noble",
    "Festive", "Gentle", "Sunny", "Whimsical", "Lively", "Radiant", "Humble", "Pure",
    "Epic", "Grand", "Regal", "Mellow", "Sophisticated", "Trendy", "Blissful", "Glorious",
    "Red", "Blue", "Green", "Yellow", "Silver", "Copper", "Bronze", "Emerald",
    "Ruby", "Sapphire", "Crimson", "Ivory"
]

SECOND_WORDS = [
    "Grill", "Cafe", "Supplies", "Boutique", "Market", "Emporium", "Bakery", "Studio",
    "Haven", "Den", "Shop", "Corner", "Workshop", "Gallery", "Bar", "Parlor",
    "Restaurant", "Depot", "House", "Lounge", "Store", "Warehouse", "Inn", "Library",
    "Arcade", "Hall", "Retreat", "Garden", "Farm", "Oasis", "Tavern", "Pavilion",
    "Salon", "Bistro", "Deli", "Mart", "Vault", "Alcove", "Forge", "Sanctuary",
    "Grove", "Terrace", "Arena", "Dock", "Station", "Hub", "Cafe", "Barbers",
    "Bakery", "Tavern", "Boutique", "Diner", "Market", "Emporium", "Studio", "Gallery",
    "Inn", "Lounge"
]


# -------------------------------------------------------------------
# Utility Functions (Pure Functions and Helpers)
# -------------------------------------------------------------------

def slight_curve_handles(curve_obj: bpy.types.Object, curviness: float = 0.1) -> None:
    """
    Adjust the bezier handles of a curve object to create a slight curvature.

    Args:
        curve_obj: The Blender curve object.
        curviness: 0.0 produces angular segments; 1.0 produces full auto curvature.
    """
    if curve_obj.type != 'CURVE':
        return

    for spline in curve_obj.data.splines:
        if spline.type == 'BEZIER':
            for bp in spline.bezier_points:
                bp.handle_left_type = 'AUTO'
                bp.handle_right_type = 'AUTO'
                auto_left = bp.handle_left.copy()
                auto_right = bp.handle_right.copy()
                bp.handle_left_type = 'FREE'
                bp.handle_right_type = 'FREE'
                bp.handle_left = bp.co + (auto_left - bp.co) * curviness
                bp.handle_right = bp.co + (auto_right - bp.co) * curviness


def point_in_polygon(point: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    """
    Determine if a 2D point is inside a polygon using the ray-casting algorithm.

    Args:
        point: (x, y) coordinates of the point.
        poly: List of (x, y) tuples defining the polygon.

    Returns:
        True if the point is inside the polygon, else False.
    """
    x, y = point
    inside = False
    n = len(poly)
    if n < 3:
        return False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def polygon_area_centroid(poly: List[Tuple[float, float]]) -> Tuple[float, Tuple[float, float]]:
    """
    Compute the area and centroid of a 2D polygon.

    Args:
        poly: List of (x, y) tuples representing the polygon vertices.

    Returns:
        A tuple containing the area and the centroid coordinates (x, y).
    """
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
    A = A / 2.0
    if A == 0:
        avg_x = sum(x for x, _ in poly) / n
        avg_y = sum(y for _, y in poly) / n
        return 0, (avg_x, avg_y)
    if A < 0:
        A = -A
        Cx = -Cx
        Cy = -Cy
    Cx = Cx / (6 * A)
    Cy = Cy / (6 * A)
    return A, (Cx, Cy)


def offset_polygon(poly: List[Tuple[float, float]], margin: float) -> List[Tuple[float, float]]:
    """
    Compute an offset (inflated) polygon from a given polygon.

    Args:
        poly: List of (x, y) tuples (closed polygon).
        margin: Distance to offset outward.

    Returns:
        A list of (x, y) tuples representing the offset polygon.
    """
    if len(poly) < 3:
        return poly.copy()

    vec_poly = [Vector(p) for p in poly]
    area, _ = polygon_area_centroid(poly)
    sign_factor = 1 if area >= 0 else -1
    effective_margin = margin if area >= 0 else -margin

    offset_poly = []
    n = len(vec_poly)
    for i in range(n):
        p_prev = vec_poly[i - 1]
        p_curr = vec_poly[i]
        p_next = vec_poly[(i + 1) % n]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        if v1.length == 0 or v2.length == 0:
            offset_poly.append((p_curr.x, p_curr.y))
            continue
        v1.normalize()
        v2.normalize()
        n1 = Vector((v1.y, -v1.x)) * sign_factor
        n2 = Vector((v2.y, -v2.x)) * sign_factor
        bisector = n1 + n2
        if bisector.length == 0:
            bisector = n1.copy()
        else:
            bisector.normalize()
        denom = bisector.dot(n1)
        if abs(denom) < 1e-6:
            offset_distance = effective_margin
        else:
            offset_distance = effective_margin / denom
        new_point = p_curr + bisector * offset_distance
        offset_poly.append((new_point.x, new_point.y))
    return offset_poly


def remove_interior_shapes(curve_obj: bpy.types.Object, area_factor: float = 1.5) -> None:
    """
    Remove interior (hole) splines from a curve object.

    Args:
        curve_obj: The Blender curve object.
        area_factor: Factor to determine if a spline is considered interior.
    """
    if curve_obj.type != 'CURVE':
        return

    spline_data = []
    for spline in curve_obj.data.splines:
        if not spline.use_cyclic_u:
            continue
        if spline.type == 'BEZIER':
            pts = [p.co for p in spline.bezier_points]
        elif spline.type == 'POLY':
            pts = [p.co for p in spline.points]
        else:
            continue

        if len(pts) < 3:
            continue

        poly = [(p.x, p.y) for p in pts]
        area, centroid = polygon_area_centroid(poly)
        spline_data.append({
            'spline': spline,
            'area': area,
            'centroid': centroid,
            'poly': poly
        })

    splines_to_remove = []
    for data in spline_data:
        area = data['area']
        centroid = data['centroid']
        candidate = data['spline']
        for other_data in spline_data:
            if candidate == other_data['spline']:
                continue
            if other_data['area'] > area * area_factor:
                if point_in_polygon(centroid, other_data['poly']):
                    splines_to_remove.append(candidate)
                    break

    for spline in splines_to_remove:
        try:
            curve_obj.data.splines.remove(spline)
        except Exception as e:
            print("Error removing spline:", e)


# -------------------------------------------------------------------
# NeonShopNameGenerator Class
# -------------------------------------------------------------------

class NeonShopNameGenerator:
    """
    A generator for creating neon shop name objects with optional outline, backing,
    dynamic scaling, rotation, and location in Blender.

    Note:
        The rotation parameter is expected in degrees. It will be converted to radians internally.
    """

    def __init__(self,
                 create_outline: bool = False,
                 curviness: float = 0.1,
                 remove_interior: bool = False,
                 create_backing: bool = True,
                 backing_margin: float = 0.2,
                 backing_bevel_depth: float = 0.02,
                 max_width: Optional[float] = None,
                 max_height: Optional[float] = None,
                 shop_name: Optional[str] = None,
                 location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize the generator with configuration parameters.

        Args:
            create_outline: Whether to duplicate the curve for an outline effect.
            curviness: Factor controlling bezier handle rounding.
            remove_interior: Whether to remove interior hole shapes.
            create_backing: Whether to create a backing outline behind the neon text.
            backing_margin: Offset distance for the backing outline.
            backing_bevel_depth: Bevel depth for the backing outline.
            max_width: Maximum allowed width (in Blender units) for the generated object.
            max_height: Maximum allowed height (in Blender units) for the generated object.
            shop_name: Optional custom shop name. If provided, it replaces the randomly generated text.
            location: The (x, y, z) location where the neon shop name will be placed.
            rotation: The (x, y, z) Euler rotation in degrees for the neon shop name.
                      (It will be converted to radians internally.)
        """
        self.create_outline = create_outline
        self.curviness = curviness
        self.remove_interior = remove_interior
        self.create_backing = create_backing
        self.backing_margin = backing_margin
        self.backing_bevel_depth = backing_bevel_depth
        self.max_width = max_width
        self.max_height = max_height
        self.shop_name = shop_name  # Custom shop text if provided.
        self.location = location
        # Convert rotation from degrees to radians.
        self.rotation = tuple(math.radians(angle) for angle in rotation)

    @staticmethod
    def generate_shop_name() -> str:
        """
        Generate a random shop name using predefined word lists.

        Returns:
            A string representing the shop name.
        """
        first_word = random.choice(FIRST_WORDS)
        second_word = random.choice(SECOND_WORDS)
        return f"{first_word} {second_word}"

    def _create_text_object(self, name: str,
                            location: Tuple[float, float, float],
                            rotation: Tuple[float, float, float]) -> bpy.types.Object:
        """
        Create a Blender text object with the given name, location, and rotation.

        Args:
            name: The text to display.
            location: The location for the text object.
            rotation: The rotation for the text object (Euler angles in radians).

        Returns:
            The created Blender text object.
        """
        bpy.ops.object.text_add(location=location, rotation=rotation)
        text_obj = bpy.context.object
        text_obj.data.body = name
        text_obj.name = name
        text_obj.data.name = name
        # Set neon-style text properties.
        text_obj.data.extrude = 0.0
        text_obj.data.bevel_depth = 0.02
        text_obj.data.bevel_resolution = 4
        text_obj.data.fill_mode = 'NONE'
        return text_obj

    @staticmethod
    def _convert_text_to_curve(text_obj: bpy.types.Object) -> bpy.types.Object:
        """
        Convert a text object to a curve object.

        Args:
            text_obj: The Blender text object.

        Returns:
            The converted curve object.
        """
        bpy.ops.object.convert(target='CURVE')
        return bpy.context.object

    @staticmethod
    def _create_neon_material() -> bpy.types.Material:
        """
        Create a neon emission material.

        Returns:
            A Blender material configured for neon emission.
        """
        material = bpy.data.materials.new(name="NeonMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        # Clear existing nodes.
        for node in list(nodes):
            nodes.remove(node)
        emission_node = nodes.new(type='ShaderNodeEmission')
        emission_node.inputs[0].default_value = (1.0, 0.2, 0.8, 1.0)  # Pink neon color.
        emission_node.inputs[1].default_value = 5.0  # Emission strength.
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(emission_node.outputs[0], output_node.inputs[0])
        return material

    @staticmethod
    def _create_outline_material() -> bpy.types.Material:
        """
        Create an outline material for the backing object.

        Returns:
            A Blender material configured for an outline appearance.
        """
        material = bpy.data.materials.new(name="OutlineMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        # Clear existing nodes.
        for node in list(nodes):
            nodes.remove(node)
        principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled_bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1.0)  # Dark gray.
        principled_bsdf.inputs['Roughness'].default_value = 0.7
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(principled_bsdf.outputs[0], output_node.inputs[0])
        return material

    @staticmethod
    def _assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
        """
        Assign a material to a Blender object.

        Args:
            obj: The Blender object.
            material: The material to assign.
        """
        if not obj.data.materials:
            obj.data.materials.append(material)
        else:
            obj.data.materials[0] = material

    def _scale_to_fit(self, obj: bpy.types.Object) -> None:
        """
        Scale the given object so that its dimensions do not exceed the max_width and max_height
        (if specified). This helps ensure that the generated neon text fits within a parent container.

        Args:
            obj: The Blender object to scale.
        """
        if self.max_width is None and self.max_height is None:
            return

        # Update the scene so that the object's dimensions are current.
        bpy.context.view_layer.update()
        dims = obj.dimensions
        scale_factors = []

        if self.max_width is not None and dims.x > self.max_width:
            scale_factors.append(self.max_width / dims.x)
        if self.max_height is not None and dims.y > self.max_height:
            scale_factors.append(self.max_height / dims.y)

        if scale_factors:
            factor = min(scale_factors)
            obj.scale *= factor
            bpy.context.view_layer.objects.active = obj

            # For 2D curves, Blender does not allow applying location or rotation.
            if obj.type == 'CURVE' and obj.data.dimensions == '2D':
                original_dimension = obj.data.dimensions
                obj.data.dimensions = '3D'
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                obj.data.dimensions = original_dimension
            else:
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    def _create_backing_outline(self, curve_obj: bpy.types.Object, shop_name: str) -> Optional[bpy.types.Object]:
        """
        Create a backing outline that traces the text path.

        Args:
            curve_obj: The primary neon curve object.
            shop_name: The shop name used for naming the outline object.

        Returns:
            The backing outline object, or None if no outline was created.
        """
        depsgraph = bpy.context.evaluated_depsgraph_get()
        curve_eval = curve_obj.evaluated_get(depsgraph)

        backing_curve_data = bpy.data.curves.new(name=f"{shop_name} Outline", type='CURVE')
        backing_curve_data.dimensions = '2D'
        backing_curve_data.fill_mode = 'NONE'
        backing_curve_data.bevel_depth = self.backing_bevel_depth

        created_spline = False
        for spline in curve_eval.data.splines:
            if not spline.use_cyclic_u:
                continue

            poly: List[Tuple[float, float]] = []
            if hasattr(spline, "evaluated_points") and len(spline.evaluated_points) >= 3:
                poly = [(pt.co.x, pt.co.y) for pt in spline.evaluated_points]
            elif spline.type == 'BEZIER' and len(spline.bezier_points) >= 3:
                poly = [(p.co.x, p.co.y) for p in spline.bezier_points]
            elif spline.type == 'POLY' and len(spline.points) >= 3:
                poly = [(p.co.x, p.co.y) for p in spline.points]
            else:
                continue

            offset_poly_points = offset_polygon(poly, self.backing_margin)
            new_spline = backing_curve_data.splines.new('POLY')
            new_spline.points.add(len(offset_poly_points) - 1)  # One point already exists.
            for i, (x, y) in enumerate(offset_poly_points):
                new_spline.points[i].co = (x, y, 0, 1)
            new_spline.use_cyclic_u = True
            created_spline = True

        if not created_spline:
            return None

        backing_obj = bpy.data.objects.new(f"{shop_name} Outline", backing_curve_data)
        bpy.context.collection.objects.link(backing_obj)
        # Set the backing object's location and rotation to match the primary curve.
        backing_obj.location = curve_obj.location.copy()
        backing_obj.location.z = curve_obj.location.z - 0.05
        backing_obj.rotation_euler = curve_obj.rotation_euler.copy()  # Copy rotation.
        self._assign_material(backing_obj, self._create_outline_material())
        return backing_obj

    def generate(self) -> Dict[str, bpy.types.Object]:
        """
        Generate the neon shop name object with optional outline, backing, rotation,
        location, and dynamic scaling to ensure it fits within a container specified by max_width and max_height.

        Returns:
            A dictionary of the created objects. For example:
                {
                    "neon": <main neon curve object>,
                    "outline": <outline object, if created>,
                    "backing": <backing object, if created>
                }
        """
        # Use the provided shop_name if given, else generate a random one.
        shop_name = self.shop_name if self.shop_name is not None else self.generate_shop_name()
        # Create text object at the specified location and rotation.
        text_obj = self._create_text_object(shop_name, self.location, self.rotation)
        # Convert text to curve.
        curve_obj = self._convert_text_to_curve(text_obj)
        # Adjust bezier handles.
        slight_curve_handles(curve_obj, self.curviness)
        if self.remove_interior:
            remove_interior_shapes(curve_obj)

        # Scale the curve object to fit within the specified max dimensions.
        self._scale_to_fit(curve_obj)

        # Prepare to assign a neon material.
        target_objects = [curve_obj]
        outline_obj = None

        if self.create_outline:
            # Duplicate curve object for an outline effect.
            bpy.ops.object.duplicate()
            outline_obj = bpy.context.object
            slight_curve_handles(outline_obj, self.curviness)
            if self.remove_interior:
                remove_interior_shapes(outline_obj)
            target_objects.append(outline_obj)

        neon_material = self._create_neon_material()
        for obj in target_objects:
            self._assign_material(obj, neon_material)

        backing_obj = None
        if self.create_backing:
            backing_obj = self._create_backing_outline(curve_obj, shop_name)
            if backing_obj:
                print(f"Generated backing outline for: {shop_name}")

        print(f"Generated and added neon shop name: {shop_name}")

        # Return a dictionary of created objects.
        result = {"neon": curve_obj}
        if outline_obj is not None:
            result["outline"] = outline_obj
        if backing_obj is not None:
            result["backing"] = backing_obj
        return result


# -------------------------------------------------------------------
# Module-Level Function
# -------------------------------------------------------------------

def generate_sign(create_outline: bool = False,
                  curviness: float = 0.1,
                  remove_interior: bool = False,
                  create_backing: bool = True,
                  backing_margin: float = 0.2,
                  backing_bevel_depth: float = 0.02,
                  max_width: Optional[float] = None,
                  max_height: Optional[float] = None,
                  shop_name: Optional[str] = None,
                  location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
                  ) -> Dict[str, bpy.types.Object]:
    """
    Generate a neon shop sign in Blender with the given parameters.

    Args:
        create_outline: Whether to duplicate the curve for an outline effect.
        curviness: Factor controlling bezier handle rounding.
        remove_interior: Whether to remove interior hole shapes.
        create_backing: Whether to create a backing outline behind the neon text.
        backing_margin: Offset distance for the backing outline.
        backing_bevel_depth: Bevel depth for the backing outline.
        max_width: Maximum allowed width (in Blender units) for the generated object.
        max_height: Maximum allowed height (in Blender units) for the generated object.
        shop_name: Optional custom shop name. If provided, it replaces the randomly generated text.
        location: The (x, y, z) location where the neon shop name will be placed.
        rotation: The (x, y, z) Euler rotation in degrees for the neon shop name.

    Returns:
        A dictionary containing the generated objects. For example:
            {
                "neon": <main neon curve object>,
                "outline": <outline object, if created>,
                "backing": <backing object, if created>
            }
    """
    generator = NeonShopNameGenerator(
        create_outline=create_outline,
        curviness=curviness,
        remove_interior=remove_interior,
        create_backing=create_backing,
        backing_margin=backing_margin,
        backing_bevel_depth=backing_bevel_depth,
        max_width=max_width,
        max_height=max_height,
        shop_name=shop_name,
        location=location,
        rotation=rotation
    )
    return generator.generate()


# -------------------------------------------------------------------
# Optional Testing Code
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Example usage: adjust parameters as desired.
    result = generate_sign(
        create_outline=False,
        curviness=0.4,
        remove_interior=True,
        create_backing=True,
        backing_margin=0.1,
        backing_bevel_depth=0.01,
        max_width=4.0,
        max_height=2.0,
        location=(1.0, 2.0, 0.0),
        rotation=(0.0, 0.0, 0.0)
    )
    print("Sign objects created:", result)
