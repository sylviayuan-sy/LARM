# . install_env.sh
# blenderproc run render_shape.py --custom-blender-path=/opt/blender-4.0.2-linux-x64 --glbdata_dir "glbs/" --object_path "dumptruck" --output_dir "./figs/"

# isort: off
import blenderproc as bproc
from blenderproc.python.utility.SetupUtility import SetupUtility
from blenderproc.python.utility.Utility import Utility
from blenderproc.scripts.saveAsImg import save_array_as_image
import bpy
import bmesh
from glob import glob

# isort: on
import argparse
import os
# Enable OpenEXR support in OpenCV, before importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import json 
from pathlib import Path
import math
import numpy as np
from mathutils import Vector, Matrix
from scipy.spatial.transform import Rotation as R
import tempfile
import random
import calibur
from calibur import CC
from urdfpy import URDF
from PIL import Image
import imageio
# imageio.plugins.freeimage.download()


def disable_all_denoiser():
    """ Disables all denoiser.

    At the moment this includes the cycles and the intel denoiser.
    """
    # Disable cycles denoiser
    bpy.context.view_layer.cycles.use_denoising = False
    bpy.context.scene.cycles.use_denoising = False

    # Disable intel denoiser
    if bpy.context.scene.use_nodes:
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        # Go through all existing denoiser nodes
        for denoiser_node in Utility.get_nodes_with_type(nodes, 'CompositorNodeDenoise'):
            in_node = denoiser_node.inputs['Image']
            out_node = denoiser_node.outputs['Image']

            # If it is fully included into the node tree
            if in_node.is_linked and out_node.is_linked:
                # There is always only one input link
                in_link = in_node.links[0]
                # Connect from_socket of the incoming link with all to_sockets of the out going links
                for link in out_node.links:
                    links.new(in_link.from_socket, link.to_socket)

            # Finally remove the denoiser node
            nodes.remove(denoiser_node)

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # r, i, j, k = np.unbind(quaternions, -1)
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = np.linalg.norm(axis_angle, ord=2, axis = -1, keepdims = True)
    # angles = axis_angle.norm(p = 2, dim = -1, keepdim = True)
    # angles = np.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def add_lighting() -> None:
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 70000
    bpy.data.objects["Area"].location[2] = 5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100

# ---------------------------------------------------------------------------- #
# Arguments
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--use_gpu", type=int, default=0)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--scale", type=float, default=1.)
parser.add_argument("--radius", type=float, default=1.)
parser.add_argument("--num_views", type=int, default=1)
parser.add_argument("--num_inputs", type=int, default=6)
parser.add_argument("--num_outputs", type=int, default=6)
parser.add_argument("--seed", type=int)
parser.add_argument("--engine", type=str, default="cycles")
parser.add_argument("--out_depth", action="store_true")
parser.add_argument("--out_normal", action="store_true")
parser.add_argument("--random", type=int, default=0)
parser.add_argument("--random_angle", type=int, default=0)
parser.add_argument("--hdri_path", type=str)
parser.add_argument("--hdri_rotation_euler", type=float)
parser.add_argument("--hdri_strength", type=float)
parser.add_argument("--joint_pose", type=int, default=0)
parser.add_argument("--joint_name", type=str, default=0)
parser.add_argument('--obj_scale', type=str, nargs="+")
parser.add_argument('--tex_no', type=int, default=0)
parser.add_argument('--color_ind', type=str)

args = parser.parse_args()

n_threads = 16
uid = args.object_path.split("/")[-2]

args.output_dir = f'{args.output_dir}/views_{uid}_{args.joint_name}_{"_".join(args.obj_scale)}_{args.tex_no}'

np.random.seed(args.seed)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------- #
# Camera settings
# ---------------------------------------------------------------------------- #

THETA = np.arctan(32/2/35)
BETA = np.arctan(np.sqrt(2))

DELTA_AZIM  = np.radians([30 + idx*60 for idx in range(args.num_outputs)])
DELTA_ELEV = np.radians([30, -30]*3)
FIXED_ELEV = np.radians([70, 100]*3)

# ---------------------------------------------------------------------------- #
# Initialize bproc
# ---------------------------------------------------------------------------- #
bproc.init()

# Renderer setting (following GET3D)
if args.engine == "cycles":
    if args.use_gpu:
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            if d["name"] == "Intel Xeon Platinum 8255C CPU @ 2.50GHz":
                d["use"] = 0
            elif d["name"] == "AMD EPYC 7502 32-Core Processor":
                d["use"] = 0
            else:
                d["use"] = 1
            print(d["name"], d["use"])
    else:
        bproc.renderer.set_render_devices(use_only_cpu=True)
        bproc.renderer.set_cpu_threads(n_threads)

    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    # breakpoint()
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.render.film_transparent = True
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.set_light_bounces(
        diffuse_bounces=1,
        glossy_bounces=1,
        transmission_bounces=3,
        transparent_max_bounces=3,
    )
    bproc.renderer.set_max_amount_of_samples(32)
else:
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    bpy.context.scene.eevee.use_raytracing = True
    bpy.context.scene.eevee.use_shadows = True
    bpy.context.scene.eevee.ray_tracing_options.use_denoise = True
    bpy.context.scene.eevee.use_bokeh_jittered = False
    bpy.context.scene.eevee.use_shadows = True
    bpy.context.scene.eevee.use_volumetric_shadows = True
    bpy.context.scene.eevee.use_raytracing = True
    bpy.context.scene.eevee.use_shadow_jitter_viewport = True
    bpy.context.scene.eevee.use_bokeh_jittered = False
    bpy.context.scene.eevee.volumetric_samples = 16
    bpy.context.scene.eevee.taa_render_samples = 16

    bproc.renderer.set_render_devices(use_only_cpu=True)
    bproc.renderer.set_cpu_threads(n_threads)

    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.render.film_transparent = True

# https://github.com/DLR-RM/BlenderProc/blob/4ca8b5f84f21a735a137d40fce366ea9e82a7822/blenderproc/python/loader/ShapeNetLoader.py
def correct_materials(obj):
    """ If the used material contains an alpha texture, the alpha texture has to be flipped to be correct

    :param obj: object where the material maybe wrong
    """
    for material in obj.material_slots:
        if material is None:
            continue
        texture_nodes = material.material.node_tree.nodes
        if texture_nodes and len(texture_nodes) > 1:
            for n in texture_nodes:
                if n.name == "Principled BSDF":
                    principled_bsdf = n

            # find the image texture node which is connected to alpha
            node_connected_to_the_alpha = None
            for node_links in principled_bsdf.inputs["Alpha"].links:
                breakpoint()
                if "ShaderNodeTexImage" in node_links.from_node.bl_idname:
                    node_connected_to_the_alpha = node_links.from_node
            # if a node was found which is connected to the alpha node, add an invert between the two
            if node_connected_to_the_alpha is not None:
                invert_node = material.new_node("ShaderNodeInvert")
                invert_node.inputs["Fac"].default_value = 1.0
                material.insert_node_instead_existing_link(node_connected_to_the_alpha.outputs["Color"],
                                                           invert_node.inputs["Color"],
                                                           invert_node.outputs["Color"],
                                                           principled_bsdf.inputs["Alpha"])
    # obj.persist_transformation_into_mesh(location=False, rotation=True, scale=False)

# ---------------------------------------------------------------------------- #
# Load model
# ---------------------------------------------------------------------------- #
try:
    fl = open(os.path.join(args.metadata_dir, uid + ".json"), "r")
    meta_load = json.load(fl)
    animation_count = meta_load[uid]["animationCount"]
except:
    animation_count = 0

joint_pose_path = args.object_path.replace("textured_objs", "joint_pose.json")

with open(joint_pose_path, "r") as f:
    scaled_joint_poses = json.load(f)

joint_poses = scaled_joint_poses[args.tex_no][args.joint_name][" ".join(args.obj_scale)]
movable_parts = scaled_joint_poses[args.tex_no][args.joint_name]["move_parts"]
joint_origin_dict = scaled_joint_poses[args.tex_no][args.joint_name]["origin"]
joint_axis = np.array(scaled_joint_poses[args.tex_no][args.joint_name]["axis"])
links = joint_poses["links"]
joint_pose_list = joint_poses["pose_list"]
pose = joint_poses["transform"]
intrinsics = np.array(scaled_joint_poses[args.tex_no]["intrinsics"])
fovy = np.array(scaled_joint_poses[args.tex_no]["fovy"])
tex_map = scaled_joint_poses[args.tex_no]["texture"]

name = f"{args.object_path.split('/')[-2]}_{args.joint_name}"

def bpy_cleanup_mesh(mesh_obj):
    obj = mesh_obj.blender_obj
    assert obj.type == 'MESH'
    # remove duplicate vertices
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=1e-06)
    bpy.ops.object.mode_set(mode='OBJECT')
    # disable auto-smoothing
    obj.data.use_auto_smooth = False
    # split edges with an angle above 70 degrees (1.22 radians)
    m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
    m.split_angle = 1.22173
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")
    # move every face an epsilon in the direction of its normal, to reduce clipping artifacts
    m = obj.modifiers.new("Displace", "DISPLACE")
    m.strength = 0.00001
    bpy.ops.object.modifier_apply(modifier="Displace")
            
objs = []
obj_names = []
for path in glob(os.path.join(args.object_path, "*.obj")):
    objs.append(bproc.loader.load_obj(
        path,
    )[0])
    obj_names.append(os.path.basename(path).split(".")[0])

for i in range(len(objs)):
    # breakpoint()
    if "textured_objs/" + obj_names[i] + ".obj" in movable_parts:
        objs[i].set_cp("category_id", 1)
    else:
        objs[i].set_cp("category_id", 0)

for obj in objs:
    bpy_cleanup_mesh(obj)

with open("./texture.txt", "r") as f:
    texture_list = f.readlines()
colors = [h.replace("\n", "") for h in texture_list]
color_ind = args.color_ind.split("_")
color_ind = [int(i) for i in color_ind]
colors = [colors[c] for c in color_ind]

if args.tex_no != 0:
    count = 0
    color_map = {}
    for mat in bpy.data.materials:
        for node in mat.node_tree.nodes:
            mat_name = mat.name.split(".")[0]
            if node.type == 'BSDF_PRINCIPLED':
                node_tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
                if mat_name not in color_map:
                    node_tex.image = bpy.data.images.load(colors[count])
                    color_map[mat_name] = colors[count]
                    count += 1
                else:
                    node_tex.image = bpy.data.images.load(color_map[mat_name])
                link = mat.node_tree.links.new(node_tex.outputs["Color"], node.inputs["Base Color"])
for mat in bproc.material.collect_all():
    mat.set_principled_shader_value('Alpha', 1.0)

# # Set the frame to the last frame of your animation
bpy.context.scene.frame_set(bpy.context.scene.frame_end)

bproc.utility.reset_keyframes()

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def get_center(l):
    return (max(l) + min(l)) / 2 if l else 0.0

def scene_sphere(single_obj=None, ignore_matrix=False):
    found = False
    points_co_global = []
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        mesh = obj.data
        for vertex in mesh.vertices:
            vertex_co = vertex.co
            if not ignore_matrix:
                vertex_co = obj.matrix_world @ vertex_co
            points_co_global.extend([vertex_co])
    if not found:
        raise RuntimeError("no objects in scene to compute bounding sphere for")
    x, y, z = [[point_co[i] for point_co in points_co_global] for i in range(3)]
    b_sphere_center = Vector([get_center(axis) for axis in [x, y, z]]) if (x and y and z) else None
    b_sphere_radius = max(((point - b_sphere_center) for point in points_co_global)) if b_sphere_center else None
    return b_sphere_center, b_sphere_radius.length

def projected_bound_circle(rotation_matrix):
    found = False
    points_co_global = []
    for obj in scene_meshes():
        found = True
        mesh = obj.data
        for vertex in mesh.vertices:
            vertex_co = vertex.co
            vertex_co = obj.matrix_world @ vertex_co
            points_co_global.extend([vertex_co])
    if not found:
        raise RuntimeError("no objects in scene to compute bounding sphere for")
    x, y, z = [[point_co[i] for point_co in points_co_global] for i in range(3)]
    if not (x and y and z):
        return None
    points_view_space = np.array(points_co_global) @ np.linalg.inv(np.array(rotation_matrix)).T
    diff = points_view_space[..., :2] - points_view_space[..., :2].mean(0)
    return calibur.magnitude(diff).max()


def add_intrinsics_frame(intrinsics, frame):
    cam_ob = bpy.context.scene.camera
    bproc.camera.set_intrinsics_from_K_matrix(intrinsics, args.resolution, args.resolution)
    if bpy.context.scene.frame_end < frame + 1:
        bpy.context.scene.frame_end = frame + 1
    cam_ob.data.keyframe_insert(data_path='shift_x', frame=frame)
    cam_ob.data.keyframe_insert(data_path='shift_y', frame=frame)
    cam_ob.data.keyframe_insert(data_path='lens', frame=frame)

def load_exr(exr_filepath):
    return cv2.cvtColor(cv2.imread(exr_filepath, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGRA2RGBA)

name_dict = {}
for obj in objs:  
    name_dict[obj.get_name()] = obj_names[objs.index(obj)]
# orient = calibur.convert_pose(np.eye(4), ("X", "Y", "Z"), ("-X", "Z", "Y"))
orient = calibur.convert_pose(np.eye(4), ("X", "Y", "Z"), ("-X", "Z", "Y"))

c = {"object" : bpy.context.scene.objects[0],
    "selected_objects" : bpy.context.scene.objects,
    "selected_editable_objects" : bpy.context.scene.objects}

if bpy.context.scene.objects:
    with bpy.context.temp_override(**c):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

origins = {}
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        origins[name_dict[obj.name]] = np.array(obj.location.copy())

for obj in objs:
    obj.set_origin(Vector([0., 0., 0.]))

origin = (0., 0., 0.)
orient = calibur.convert_pose(np.eye(4), ("X", "Y", "Z"), ("-X", "-Z", "-Y"))
for obj in objs:  
    if obj_names[objs.index(obj)] not in pose:
        obj.set_scale([float(args.obj_scale[0]), float(args.obj_scale[1]), float(args.obj_scale[2])])
        obj.set_origin(Vector([0., 0., 0.]))
        print(obj_names[0], pose.keys())
        T = orient @ np.array(pose[obj_names[0]][0])
        T[:3, 3] *= np.array(obj.get_scale())
        obj.apply_T(T)
        scale = obj.get_scale().copy()
        obj.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
        obj.set_origin(Vector([0., 0., 0.]))
        continue
    T0 = np.array(pose[obj_names[objs.index(obj)]][-1])
    T = np.array(pose[obj_names[objs.index(obj)]][args.joint_pose])
    obj.set_scale(np.array(args.obj_scale).astype(float)) # * joint_axis * np.array([-1., -1., -1.]))
    scale = obj.get_scale().copy()
    obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
    if (np.round(T[:3, :3], decimals=6) != np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])).any():
        obj.set_origin(Vector([0., 0., 0.]))
        obj.apply_T(T0)
        scale = obj.get_scale().copy()
        obj.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
        try:
            joint_origin = np.array(joint_origin_dict[obj_names[objs.index(obj)]])
        except:
            joint_origin = np.array(joint_origin_dict[list(joint_origin_dict.keys())[0]])
        joint_origin = joint_origin
        origin = obj.set_origin(np.array([joint_origin[2], joint_origin[1], joint_origin[0]]) * np.array([args.obj_scale[2], args.obj_scale[1], args.obj_scale[0]]).astype(float))
        rotate = T.copy()
        rotate[:3, 3] = 0.
        orient_rot = np.array([[ -1.,  0.,  0.,  0.], [ 0.,  0.,  1.,  0.], [ 0.,  -1.,  0.,  0.], [ 0.,  0.,  0.,  1.]])
        copy_rot = rotate.copy()
        rotate[:, 0] = copy_rot[:, 2]
        rotate[:, 1] = copy_rot[:, 1]
        rotate[:, 2] = copy_rot[:, 0]
        # print(args.joint_pose)
        obj.apply_T(orient_rot @ rotate)
        obj.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
    else:
        obj.set_origin(Vector([0., 0., 0.]))
        T[:3, 3] *= np.array([float(args.obj_scale[2]), float(args.obj_scale[1]), float(args.obj_scale[0])])
        obj.apply_T(T)
        obj.persist_transformation_into_mesh(location=True, rotation=True, scale=True)
    obj.set_origin(Vector([0., 0., 0.]))

for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
    else:
        ob.select_set(False)
    
# normalized into a sphere with radius 0.5
center, sphere_radius = scene_sphere()
scale = 0.5 / sphere_radius

if args.joint_pose != 0:
    with open(os.path.join(output_dir, f"meta_{joint_pose_list[0]:.2f}.json"), "r") as f:
        meta_load = json.load(f)
        # breakpoint()
    center = np.array(meta_load["scale_center"])
    scale = meta_load["scale"]

for obj in scene_root_objects():
    obj.scale = obj.scale * scale * args.scale
bpy.context.view_layer.update()

new_center, _ = scene_sphere()

if args.joint_pose != 0:
    with open(os.path.join(output_dir, f"meta_{joint_pose_list[0]:.2f}.json"), "r") as f:
        meta_load = json.load(f)
    new_center = Vector(np.array(meta_load["center"]))

offset = -new_center
for obj in scene_root_objects():
    obj.matrix_world.translation += offset

bpy.ops.object.mode_set(mode = 'OBJECT')
my_specific_objects_list = bpy.data.objects
for o in my_specific_objects_list:
    o.select_set(True)
bpy.ops.object.transform_apply(scale=True)

bpy.context.view_layer.update()

IS_DEBUG = False

if IS_DEBUG:
    # add a debug sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0, 0, 0))

# ---------------------------------------------------------------------------- #
# Render
# ---------------------------------------------------------------------------- #

# Set rendering parameters
fovy_target = np.radians(30)
out_radius = 0.5 / np.tan(fovy_target / 2)
bproc.camera.set_intrinsics_from_blender_params(fovy_target, lens_unit="FOV")
bproc.camera.set_resolution(args.resolution, args.resolution)


f = calibur.fov_to_focal(fovy_target, args.resolution)
out_intrinsics = calibur.intrinsic_cv((args.resolution - 1) / 2, (args.resolution - 1) / 2, f, f)

bbox_min, bbox_max = scene_bbox()
# breakpoint()
aabb = [np.array(bbox_min).tolist(), np.array(bbox_max).tolist()]

meta = {"uid": uid, "scale": scale, "scale_center": np.array(center).tolist(), "center": np.array(new_center).tolist(), "resolution": args.resolution, "fovy": fovy_target,  "bbox": aabb, "out_radius": out_radius, "out_intrinsics": out_intrinsics.tolist(), "joint_name": args.joint_name, "joint_pose": joint_pose_list[args.joint_pose], "obj_scale": [float(s) for s in args.obj_scale], "tex_map": tex_map}

N_INPUT = args.num_inputs
N_OUTPUT = args.num_outputs

poi = np.zeros(3)
for i in range(args.num_views):
    bproc.utility.reset_keyframes()
    meta_sample = {}
    frames = []
    
    # Randomize hdri from the given haven directory as background
    haven_hdri_path = args.hdri_path
    HDRI_id = haven_hdri_path.split('/')[-1].split('.')[0]
    rand_strength = args.hdri_strength
    print("Using HDRI:", HDRI_id, "with strength =", rand_strength)
    rand_rotation_euler = [0, args.hdri_rotation_euler, 0]
    bproc.world.set_world_background_hdr_img(haven_hdri_path, strength=rand_strength, rotation_euler=rand_rotation_euler)
    hdri_info = dict(hdri_id=HDRI_id, strength=rand_strength, rotation_euler=rand_rotation_euler)
    meta_sample["hdri"] = hdri_info
    
    meta_sample["intrinsics"] = intrinsics.tolist() 

    # Add input frames
    for input_idx in range(N_INPUT):
        # Sample random camera location above objects
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.radians(np.random.uniform(0, 180))
        if input_idx == 0:
            ref_azim = azimuth
        x = np.cos(azimuth) * np.sin(elevation)
        y = np.sin(azimuth) * np.sin(elevation)
        z = np.cos(elevation)
        t_unit_vec = np.array([x, y, z])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - t_unit_vec)

        circ_rad = projected_bound_circle(rotation_matrix)
        radius_scale = np.clip(np.random.uniform(0.8, 1.7), 0.85, np.random.uniform(1.0, 1.5))
        sample_radius = circ_rad / np.tan(fovy / 2) * radius_scale
        # sample_radius = 1.0
        viewport_center = np.array([args.resolution / 2, args.resolution / 2, 1]) * sample_radius
        viewspace_center = np.linalg.inv(intrinsics) @ viewport_center
        rot_cv = np.eye(4)
        rot_cv[:3, :3] = rotation_matrix
        t = poi - calibur.transform_point(viewspace_center, calibur.convert_pose(rot_cv, CC.Blender, CC.CV))

        # Add homogeneous cam pose based on location (t) an rotation (R)
        cam2world_matrix = bproc.math.build_transformation_mat(t, rotation_matrix)

        bproc.camera.add_camera_pose(orient @ cam2world_matrix, frame=input_idx)
        add_intrinsics_frame(intrinsics, frame=input_idx)

        frame = dict(
            radius=sample_radius,
            transform_matrix=cam2world_matrix.tolist(),
            azimuth=azimuth,
            elevation=elevation,
            camera_angle_x = bpy.data.objects["Camera"].data.angle_x
        )
        meta_sample[f"input_frame_{input_idx}"] = frame

    # Add output frames
    for delta_idx in range(N_OUTPUT):
        azimuth_ = (ref_azim + DELTA_AZIM[delta_idx]) % (2*np.pi)
        elevation_ = FIXED_ELEV[delta_idx]

        x = np.cos(azimuth_) * np.sin(elevation_)
        y = np.sin(azimuth_) * np.sin(elevation_)
        z = np.cos(elevation_)
        location = np.array([x, y, z]) * out_radius

        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

        # Add homogeneous cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        bproc.camera.add_camera_pose(orient @ cam2world_matrix, frame=N_INPUT+delta_idx)
        add_intrinsics_frame(out_intrinsics, frame=N_INPUT+delta_idx)

        frame = dict(
            transform_matrix=cam2world_matrix.tolist(),
            azimuth=azimuth_,
            elevation=elevation_,
        )
        frames.append(frame)
    meta_sample["view_frames"] = frames
    meta[f"sample_{i}"] = meta_sample

    # Render images
    bproc.renderer.set_cpu_threads(n_threads)
    # Output settings
    if args.out_depth:
        bproc.renderer.enable_depth_output(False, output_dir=str(os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}")), file_prefix=f"depth_")
    if args.out_normal:
        bproc.renderer.enable_normals_output(output_dir=str(os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}")), file_prefix="normal_")
    data = bproc.renderer.render(output_dir=str(os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}")), return_data=False)

    data.update(bproc.renderer.render_segmap(output_dir=str(os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}")),
        map_by=["class"],
        file_prefix='mask_', 
        output_key='mask', 
        use_alpha_channel=True))
    
    for index in range(N_INPUT + N_OUTPUT):
        source_fn = os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}", f"mask_{index:04d}.npy")
        if index < N_INPUT:
            target_fn = os.path.join(output_dir, f"mask_{joint_pose_list[args.joint_pose]:.2f}_in_{index:02d}.png")
        else:
            target_fn = os.path.join(output_dir, f"mask_{joint_pose_list[args.joint_pose]:.2f}_out_{index-N_INPUT:02d}.png")
        mask = np.load(source_fn)
        Image.fromarray(mask * 255).save(target_fn)
        os.remove(source_fn)

    for index in range(N_INPUT + N_OUTPUT):
        source_fn = os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}", f"rgb_{index:04d}.png")
        if index < N_INPUT:
            target_fn = os.path.join(output_dir, f"color_{joint_pose_list[args.joint_pose]:.2f}_in_{index:02d}.png")
        else:
            target_fn = os.path.join(output_dir, f"color_{joint_pose_list[args.joint_pose]:.2f}_out_{index-N_INPUT:02d}.png")
        os.rename(source_fn, target_fn)

    if args.out_normal:
        for index in range(N_INPUT + N_OUTPUT):
            source_fn = os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}", f"normal_{index:04d}.exr")
            if index < N_INPUT:
                target_fn = os.path.join(output_dir, f"normal_{joint_pose_list[args.joint_pose]:.2f}_in_{index:02d}.exr")
            else:
                target_fn = os.path.join(output_dir, f"normal_{joint_pose_list[args.joint_pose]:.2f}_out_{index-N_INPUT:02d}.exr")
            os.rename(source_fn, target_fn)
    if args.out_depth:
        for index in range(N_INPUT + N_OUTPUT):
            source_fn = os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}", f"depth_{index:04d}.exr")
            if index < N_INPUT:
                target_fn = os.path.join(output_dir, f"depth_{joint_pose_list[args.joint_pose]:.2f}_in_{index:02d}.exr")
            else:
                target_fn = os.path.join(output_dir, f"depth_{joint_pose_list[args.joint_pose]:.2f}_out_{index-N_INPUT:02d}.exr")
            os.rename(source_fn, target_fn)
    os.removedirs(os.path.join(output_dir, f"{joint_pose_list[args.joint_pose]:.2f}"))


dumpmeta_dir = os.path.join(output_dir, f"meta_{joint_pose_list[args.joint_pose]:.2f}.json")
with open(dumpmeta_dir, "w") as f:
    json.dump(meta, f, indent=4)
