"""Render armor scenes with Blender: blender -b -t 8 --python synthetic/render.py -- --help."""
import argparse
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector

p = argparse.ArgumentParser()
p.add_argument('--assets', required=True, type=Path)
p.add_argument('--sim', required=True, type=Path, help='Read-only sim checkout containing sentry and field CAD')
p.add_argument('--hdri', required=True, type=Path, help='Directory containing at least two HDR environment maps')
p.add_argument('--output', required=True, type=Path)
p.add_argument('--count', type=int, default=100)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--heldout', action='store_true', help='Use distinct background and lighting ranges')
p.add_argument('--faces', choices=['sticker', 'mixed', 'matrix'], default='mixed')
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
rng = random.Random(a.seed)
a.output.mkdir(parents=True, exist_ok=True)
for sub in ('images', 'labels', 'truth'): (a.output / sub).mkdir(exist_ok=True)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
s = bpy.context.scene
s.render.engine = 'CYCLES'
s.cycles.samples = 8
s.cycles.use_denoising = False
s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = 640, 480, 100
s.render.image_settings.file_format = 'JPEG'
s.render.image_settings.quality = 88
s.render.use_motion_blur = True
s.render.motion_blur_shutter = .45
s.use_nodes = True
nodes = s.node_tree.nodes; nodes.clear()
layer = nodes.new('CompositorNodeRLayers'); glare = nodes.new('CompositorNodeGlare'); glare.glare_type = 'FOG_GLOW'; glare.quality = 'MEDIUM'; glare.threshold = 1
out = nodes.new('CompositorNodeComposite'); s.node_tree.links.new(layer.outputs['Image'], glare.inputs['Image']); s.node_tree.links.new(glare.outputs['Image'], out.inputs['Image'])
s.world.use_nodes = True
s.view_settings.view_transform = 'Standard'
bpy.ops.object.camera_add()
cam = bpy.context.object
cam.data.sensor_fit = 'HORIZONTAL'
cam.data.sensor_width = 36
cam.data.lens = 36 / (2 * math.tan(math.radians(69 / 2)))
s.camera = cam

# Raw FreeCAD axes differ between the small and large modules.
meshes = []
for name in ('AM02', 'AM12'):
    bpy.ops.wm.obj_import(filepath=str(a.assets / 'CAD' / (name + '.obj')), forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[0]
    for v in obj.data.vertices:
        x, y, z = v.co
        if name == 'AM02': v.co = ((z + 45) / 1000, (y - 66) / 1000, (x + 7.907) / 1000)
        else: v.co = ((z + 45) / 1000, (x + 96.5) / 1000, (y - 226.282) / 1000)
    meshes.append(obj.data)
    bpy.data.objects.remove(obj, do_unlink=True)


def material(name, color, emission=0):
    m = bpy.data.materials.new(name); m.diffuse_color = (*color, 1)
    m.use_nodes = True
    bs = m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*color, 1)
    bs.inputs['Emission Color'].default_value = (*color, 1)
    bs.inputs['Emission Strength'].default_value = emission
    return m

frame_mat = material('frame', (.08, .08, .08))
bar_mats = [material('blue', (.005, .04, 1), 25), material('red', (1, .005, .005), 25)]
body_mats = [material(str(i), col) for i, col in enumerate(((.14,.15,.18),(.015,.015,.015),(.18,.02,.02),(.01,.03,.2)))]
def textured_material(name, color1, color2, scale):
    m = material(name, color1)
    nodes, links = m.node_tree.nodes, m.node_tree.links
    noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = scale
    noise.inputs['Detail'].default_value = 3
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (*color1, 1)
    ramp.color_ramp.elements[1].color = (*color2, 1)
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], nodes.get('Principled BSDF').inputs['Base Color'])
    return m

surface_mats = [textured_material(f'surface-{i}', *colors, scale) for i, (colors, scale) in enumerate((
    (((.025,.03,.035), (.25,.24,.21)), 8),
    (((.025,.02,.015), (.22,.12,.07)), 12),
    (((.02,.04,.035), (.12,.22,.17)), 18),
    (((.02,.025,.05), (.14,.17,.24)), 26),
    (((.04,.025,.035), (.24,.13,.18)), 35),
    (((.025,.035,.04), (.17,.22,.25)), 48),
))]
face_mats = {}
for identity, file in enumerate(('1.png', '2.png', '3.png', 'Sentry.png', '4.png')):
    for color in range(2):
        for matrix in (False, True):
            m = bpy.data.materials.new(f'{identity}-{color}-{matrix}'); m.use_nodes = True
            nodes = m.node_tree.nodes; links = m.node_tree.links; bs = nodes.get('Principled BSDF')
            tex = nodes.new('ShaderNodeTexImage'); tex.image = bpy.data.images.load(str(a.assets / 'image' / file), check_existing=True)
            if matrix:
                # Quantise the texture coordinates to form a dot-matrix stand-in.
                tex.interpolation = 'Closest'
                coord = nodes.new('ShaderNodeTexCoord'); scale = nodes.new('ShaderNodeVectorMath'); scale.label = 'grid_scale'; scale.operation = 'SCALE'; scale.inputs[3].default_value = 24
                floor = nodes.new('ShaderNodeVectorMath'); floor.operation = 'FLOOR'
                inv = nodes.new('ShaderNodeVectorMath'); inv.label = 'grid_inverse'; inv.operation = 'SCALE'; inv.inputs[3].default_value = 1 / 24
                links.new(coord.outputs['UV'], scale.inputs[0]); links.new(scale.outputs[0], floor.inputs[0]); links.new(floor.outputs[0], inv.inputs[0]); links.new(inv.outputs[0], tex.inputs['Vector'])
            links.new(tex.outputs['Color'], bs.inputs['Base Color'])
            if matrix:
                frac = nodes.new('ShaderNodeVectorMath'); frac.operation = 'FRACTION'
                delta = nodes.new('ShaderNodeVectorMath'); delta.operation = 'SUBTRACT'; delta.inputs[1].default_value = (.5,.5,0)
                dot = nodes.new('ShaderNodeVectorMath'); dot.operation = 'LENGTH'
                mask = nodes.new('ShaderNodeMath'); mask.operation = 'LESS_THAN'; mask.inputs[1].default_value = .34
                links.new(scale.outputs[0], frac.inputs[0]); links.new(frac.outputs[0], delta.inputs[0]); links.new(delta.outputs[0], dot.inputs[0]); links.new(dot.outputs['Value'], mask.inputs[0])
                masked = nodes.new('ShaderNodeMixRGB'); masked.blend_type = 'MULTIPLY'; masked.inputs[0].default_value = 1
                links.new(tex.outputs['Color'], masked.inputs[1]); links.new(mask.outputs[0], masked.inputs[2])
                tint = nodes.new('ShaderNodeMixRGB'); tint.blend_type = 'MULTIPLY'; tint.inputs[0].default_value = 1
                tint.inputs[2].default_value = ((.25,.45,1,1) if color == 0 else (1,.25,.25,1))
                links.new(masked.outputs[0], tint.inputs[1]); links.new(tint.outputs[0], bs.inputs['Base Color']); links.new(tint.outputs[0], bs.inputs['Emission Color']); bs.inputs['Emission Strength'].default_value = 1.2
            face_mats[identity, color, matrix] = m


def cube(name, location, scale, mat, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    o = bpy.context.object; o.name = name; o.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    if parent: o.parent = parent
    return o


def project(point):
    v = world_to_camera_view(s, cam, point)
    return (v.x * 640, (1-v.y) * 480)

# STL coordinates are in the URDF/field Z-up frame and already in metres.
def load_stl(path):
    bpy.ops.wm.stl_import(filepath=str(path))
    obj = bpy.context.selected_objects[0]
    data = obj.data
    data.use_fake_user = True
    bpy.data.objects.remove(obj, do_unlink=True)
    return data


robot_dir = a.sim / 'urdf/sentry_v2'
robot_names = ['root', 'head', 'head_pitch'] + [f'{part}_{i}' for part in ('carrier', 'wheel') for i in range(4)]
robot_meshes = {name: load_stl(robot_dir / 'meshes' / f'{name}.stl') for name in robot_names}
urdf = ET.parse(robot_dir / 'sentry_v2.urdf').getroot()
joints = {}
for joint in urdf.findall('joint'):
    origin = joint.find('origin')
    if origin is not None:
        xyz = tuple(float(v) for v in origin.get('xyz', '0 0 0').split())
        rpy = tuple(float(v) for v in origin.get('rpy', '0 0 0').split())
        joints[joint.get('name')] = (xyz, rpy)


def part(name, data, parent, location=(0, 0, 0), rotation=(0, 0, 0), mat=None):
    obj = bpy.data.objects.new(name, data); bpy.context.collection.objects.link(obj)
    obj.parent = parent; obj.location = location; obj.rotation_euler = rotation
    obj['dynamic'] = True
    if mat:
        if not data.materials: data.materials.append(mat)
        obj.material_slots[0].link = 'OBJECT'; obj.material_slots[0].material = mat
    return obj


def empty(name, parent=None, location=(0, 0, 0), rotation=(0, 0, 0)):
    obj = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(obj)
    obj.parent = parent; obj.location = location; obj.rotation_euler = rotation
    obj['dynamic'] = True
    return obj


arena_mats = {
    'floor': material('arena floor', (.23,.22,.20)),
    'platform': material('arena platform', (.28,.28,.27)),
    'pole': material('arena poles', (.19,.19,.18)),
    'wood': material('arena wood', (.29,.18,.1)),
    'bumpy_road': material('arena road', (.19,.19,.18)),
    'dark': material('arena dark', (.05,.05,.05)),
    'tape_red': material('red tape', (.6,.02,.02)),
    'tape_blue': material('blue tape', (.02,.04,.6)),
    'tape_green': material('green tape', (.02,.5,.03)),
}
arena = []
for name, mat in arena_mats.items():
    data = load_stl(a.sim / 'world/arcc_s2_3v3/meshes' / f'visual_{name}.stl')
    obj = bpy.data.objects.new(f'arena-{name}', data); bpy.context.collection.objects.link(obj)
    data.materials.append(mat); arena.append(obj)

hdri_images = [bpy.data.images.load(str(path), check_existing=True) for path in sorted(a.hdri.glob('*.hdr'))] if a.hdri else []
if len(hdri_images) < 2:
    raise ValueError('--hdri must contain at least two .hdr files for separate held-out backgrounds')
world_nodes = s.world.node_tree.nodes
world_links = s.world.node_tree.links
background = world_nodes.get('Background')
env = world_nodes.new('ShaderNodeTexEnvironment')
metal = [material('metal grey', (.12,.13,.14)), material('metal black', (.025,.027,.03)),
         textured_material('carbon', (.012,.012,.012), (.09,.09,.09), 80)]
shell = [material('team blue shell', (.015,.035,.24)), material('team red shell', (.25,.015,.012))]
# Panel local X is horizontal, Y is up and Z is outward; the URDF armor X points outward.
panel_rotation = Matrix(((0,0,1), (1,0,0), (0,1,0))).to_4x4().to_euler()
cam.location = (0, -3.8, .43)
cam.rotation_euler = Vector((0, 1, -.03)).to_track_quat('-Z', 'Y').to_euler()


def camera_point(point):
    v = cam.matrix_world.inverted() @ point
    return [v.x, -v.y, -v.z]


def belongs_to(obj, ancestor):
    while obj:
        if obj == ancestor: return True
        obj = obj.parent
    return False


instance_index = 0
for index in range(a.count):
    for obj in list(bpy.data.objects):
        if obj.get('dynamic'): bpy.data.objects.remove(obj, do_unlink=True)
    for data in list(bpy.data.meshes):
        if data.users == 0 and data not in meshes: bpy.data.meshes.remove(data)
    for action in list(bpy.data.actions):
        if action.users == 0: bpy.data.actions.remove(action)
    for link in list(background.inputs['Color'].links): world_links.remove(link)
    alternate = bool(hdri_images) and index % 4 == 0
    if alternate:
        env.image = hdri_images[-1] if a.heldout else hdri_images[0]
        world_links.new(env.outputs['Color'], background.inputs['Color'])
        floor = cube('alternate floor', (0,0,-.04), (20,20,.08), rng.choice(surface_mats[4:] if a.heldout else surface_mats[:4])); floor['dynamic'] = True
    else:
        background.inputs['Color'].default_value = (.12,.12,.12,1)
    background.inputs['Strength'].default_value = rng.uniform(.15,.4)
    for obj in arena: obj.hide_render = alternate; obj.hide_viewport = alternate
    grid = rng.choice((34, 40) if a.heldout else (16, 20, 24, 28))
    for mat in face_mats.values():
        for node in mat.node_tree.nodes:
            if node.label == 'grid_scale': node.inputs[3].default_value = grid
            elif node.label == 'grid_inverse': node.inputs[3].default_value = 1 / grid
    s.view_settings.exposure = rng.uniform(-1.6, -.8) if a.heldout else rng.uniform(-1.4, -.6)
    for mat in bar_mats:
        mat.node_tree.nodes.get('Principled BSDF').inputs['Emission Strength'].default_value = rng.uniform(12,55)
    s.render.image_settings.quality = rng.randint(55,98)
    for location, energy, color in (((0,0,5), rng.uniform(300,600), (1,1,1)),
                                    ((rng.uniform(-4,4),0,3), rng.uniform(40,100), rng.choice(((1,.25,.2),(.2,.3,1))))):
        bpy.ops.object.light_add(type='AREA', location=location)
        light = bpy.context.object; light['dynamic'] = True; light.data.energy = energy
        light.data.color = color; light.data.size = 5
    labels, truth, pending = [], [], []
    count = rng.choices((1,2,3), (3,5,2))[0]
    for j in range(count):
        cls = ((instance_index // 2) % 4) + 4 * (instance_index % 2)
        instance_index += 1
        color, identity = divmod(cls, 4)
        identity = 4 if identity == 2 and rng.random() < .5 else identity
        matrix = a.faces == 'matrix' or (a.faces == 'mixed' and rng.random() < .5)
        distance = rng.uniform(1.5, 4) if rng.random() < .8 else rng.uniform(1, 6)
        x = rng.uniform(-.3,.3) * distance
        robot = empty('robot', location=(x, -3.8 + math.sqrt(distance**2 - x**2), 0))
        variant = rng.randrange(3)
        robot.scale = (rng.uniform(.9,1.15) if variant == 1 else rng.uniform(.95,1.05),
                       rng.uniform(.9,1.1), rng.uniform(.95,1.05))
        yaw = rng.uniform(-math.pi,math.pi); spin = rng.uniform(-.12,.12)
        for frame, offset in ((0,-1),(1,0),(2,1)):
            robot.rotation_euler.z = yaw + offset * spin
            robot.keyframe_insert(data_path='rotation_euler', frame=frame)
        s.frame_set(1)
        part('chassis', robot_meshes['root'], robot, mat=rng.choice(metal + [shell[color]]))
        head_xyz, head_rpy = joints['headlink']
        head = empty('head yaw', robot, head_xyz, head_rpy); head.rotation_euler.z = rng.uniform(-math.pi,math.pi)
        if variant == 2: head.scale.z = rng.uniform(1.05,1.15)
        part('gimbal head', robot_meshes['head'], head, mat=rng.choice(metal + [shell[color]]))
        pitch_xyz, pitch_rpy = joints['headpitch']
        pitch = empty('head pitch', head, pitch_xyz, pitch_rpy); pitch.rotation_euler.y = rng.uniform(-.3,.3)
        part('barrel', robot_meshes['head_pitch'], pitch, mat=rng.choice(metal))
        for i in range(4):
            xyz, rpy = joints[f'suspension_{i}']
            carrier = empty(f'carrier-{i}', robot, xyz, rpy)
            part('carrier', robot_meshes[f'carrier_{i}'], carrier, mat=rng.choice(metal))
            wheel = part('wheel', robot_meshes[f'wheel_{i}'], carrier, mat=metal[1])
            wheel.rotation_euler[1 - i % 2] = rng.uniform(-math.pi,math.pi)
        led = cube('referee indicator', (0,.16,.30), (.20,.015,.012), bar_mats[color], robot); led['dynamic'] = True
        for i in range(4):
            xyz, rpy = joints[f'armor_{i}link']
            mount = empty('armor mount', robot, xyz, rpy)
            panel = empty('panel', mount); panel.rotation_euler = panel_rotation
            part('armor CAD', meshes[0], panel, mat=frame_mat)
            spacing, length = .1258, .060
            for side in (-1,1):
                bar = cube('bar', (side*spacing/2,0,.004), (.010,length,.008), bar_mats[color], panel); bar['dynamic'] = True
            bpy.ops.mesh.primitive_plane_add(size=1)
            face = bpy.context.object; face.parent = panel; face.location = (0,0,.002)
            face.scale = (spacing-.012,.115,1); face.data.materials.append(face_mats[identity,color,matrix]); face['dynamic'] = True
            pending.append((cls, color, matrix, panel, spacing, length))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for cls, color, matrix, panel, spacing, length in pending:
        pos = panel.matrix_world @ Vector((0,0,.008))
        normal = (panel.matrix_world.to_3x3().inverted().transposed() @ Vector((0,0,1))).normalized()
        if normal.dot(cam.location - pos) <= 0: continue
        pts = [panel.matrix_world @ Vector((x,y,.008)) for x,y in ((-spacing/2,length/2),(spacing/2,length/2),(spacing/2,-length/2),(-spacing/2,-length/2))]
        uv = [project(point) for point in pts]
        if not all(0 <= x < 640 and 0 <= y < 480 for x,y in uv): continue
        paired = sorted(zip(uv, pts), key=lambda pair: pair[0][1])
        paired = sorted(paired[:2], key=lambda pair: pair[0][0]) + sorted(paired[2:], key=lambda pair: pair[0][0], reverse=True)
        visibility = []
        for _, point in paired:
            ray = point - cam.location
            hit, _, _, _, obj, _ = s.ray_cast(depsgraph, cam.location, ray.normalized(), distance=ray.length - .001)
            visibility.append(0 if hit and not belongs_to(obj, panel) else 2)
        if all(vis == 0 for vis in visibility): continue
        corners = [project(panel.matrix_world @ Vector((side*spacing/2+dx, y, .008)))
                   for side in (-1,1) for dx in (-.005,.005) for y in (-length/2,length/2)]
        xs, ys = zip(*corners); x0, x1 = max(0,min(xs)), min(640,max(xs)); y0, y1 = max(0,min(ys)), min(480,max(ys))
        labels.append(' '.join(map(str, [cls,(x0+x1)/1280,(y0+y1)/960,(x1-x0)/640,(y1-y0)/480]+[v for (x,y),vis in zip([p for p,_ in paired],visibility) for v in (x/640,y/480,vis)])))
        normal_cam = cam.matrix_world.to_quaternion().inverted() @ normal
        truth.append(dict(class_id=cls, colour=('blue','red')[color], face='matrix' if matrix else 'sticker', large=False,
                          keypoints_camera=[camera_point(point) for _,point in paired], visibility=visibility,
                          position=camera_point(pos), normal=[normal_cam.x,-normal_cam.y,-normal_cam.z],
                          spacing=(pts[1]-pts[0]).length, length=(pts[3]-pts[0]).length))
    stem = f'{index:06d}'
    (a.output/'labels'/f'{stem}.txt').write_text('\n'.join(labels)+'\n' if labels else '')
    (a.output/'truth'/f'{stem}.json').write_text(json.dumps(dict(width=640,height=480,hfov_degrees=69,panels=truth),indent=2)+'\n')
    s.render.filepath = str(a.output/'images'/f'{stem}.jpg')
    bpy.ops.render.render(write_still=True)
