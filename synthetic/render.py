"""Render armor scenes with Blender: blender -b -t 8 --python synthetic/render.py -- --help."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

p = argparse.ArgumentParser()
p.add_argument('--assets', required=True, type=Path)
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

instance_index = 0
for index in range(a.count):
    for o in list(bpy.data.objects):
        if o not in (cam,) and o.get('dynamic'): bpy.data.objects.remove(o, do_unlink=True)
    for data in list(bpy.data.meshes):
        if data.users == 0 and data not in meshes: bpy.data.meshes.remove(data)
    for action in list(bpy.data.actions):
        if action.users == 0: bpy.data.actions.remove(action)
    bg = (rng.uniform(.01,.06), rng.uniform(.08,.18), rng.uniform(.18,.35)) if a.heldout else tuple(rng.uniform(.02,.3) for _ in range(3))
    s.world.node_tree.nodes['Background'].inputs['Color'].default_value = (*bg,1)
    grid = rng.choice((34, 40) if a.heldout else (16, 20, 24, 28))
    for mat in face_mats.values():
        for node in mat.node_tree.nodes:
            if node.label == 'grid_scale': node.inputs[3].default_value = grid
            elif node.label == 'grid_inverse': node.inputs[3].default_value = 1 / grid
    s.view_settings.look = 'None'
    s.view_settings.exposure = rng.uniform(-1.6, -.8) if a.heldout else rng.uniform(-1.4, -.6)
    for mat in bar_mats:
        mat.node_tree.nodes.get('Principled BSDF').inputs['Emission Strength'].default_value = rng.uniform(12, 55)
    s.render.image_settings.quality = rng.randint(55, 98)
    bpy.ops.object.light_add(type='AREA', location=(rng.uniform(-3,3),rng.uniform(-2,3),rng.uniform(-1,1)))
    light = bpy.context.object; light['dynamic'] = True; light.data.energy = rng.uniform(55, 100) if a.heldout else rng.uniform(3, 50); light.data.size = rng.uniform(2,6)
    surfaces = surface_mats[4:] if a.heldout else surface_mats[:4]
    floor = cube('floor', (0,-.5,-5), (20,.05,20), rng.choice(surfaces)); floor['dynamic'] = True
    wall = cube('wall', (0,2,-9), (20,5,.1), rng.choice(surfaces)); wall['dynamic'] = True
    for _ in range(rng.randint(8, 20)):
        z = -rng.uniform(4, 9)
        height = rng.uniform(.1, 1.8)
        obj = cube('clutter', (rng.uniform(-6,6),-.5+height/2,z),
                   (rng.uniform(.1,1.5),height,rng.uniform(.1,1.2)),
                   rng.choice(body_mats + surfaces)); obj['dynamic'] = True
    count = rng.choices((0,1,2,3,4), (1,4,4,2,1))[0]
    labels, truth, pending = [], [], []
    for j in range(count):
        cls = (instance_index + a.seed) % 8
        instance_index += 1
        color, identity = divmod(cls, 4)
        identity = 4 if identity == 2 and rng.random() < .5 else identity
        matrix = a.faces == 'matrix' or (a.faces == 'mixed' and rng.random() < .5)
        large = rng.random() < .2
        distance = rng.uniform(1.5, 4) if rng.random() < .8 else rng.uniform(1, 6)
        bpy.ops.object.empty_add(location=(rng.uniform(-.35,.35)*distance, -.5 + rng.uniform(.10,.35), -distance))
        root = bpy.context.object; root['dynamic'] = True
        center = root.location.copy(); delta = Vector((rng.uniform(-.012,.012)*distance, rng.uniform(-.008,.008)*distance, 0))
        for frame, offset in ((0,-1),(1,0),(2,1)):
            root.location = center + delta*offset; root.keyframe_insert(data_path='location',frame=frame)
        s.frame_set(1)
        root.rotation_euler = (math.radians(15), math.radians(rng.uniform(-75,75)), math.radians(rng.uniform(-8,8)))
        mesh = bpy.data.objects.new('CAD', meshes[large]); bpy.context.collection.objects.link(mesh); mesh.parent = root; mesh['dynamic'] = True
        if not mesh.data.materials: mesh.data.materials.append(frame_mat)
        spacing = .216 if large else .1258
        length = .060
        for side in (-1,1):
            bar = cube('bar', (side*spacing/2,0,.004), (.010,length,.008), bar_mats[color], root); bar['dynamic'] = True
        bpy.ops.mesh.primitive_plane_add(size=1)
        face = bpy.context.object; face.parent = root; face.location = (0,0,.002); face.scale = (spacing-.012, .105 if large else .115, 1); face.data.materials.append(face_mats[identity,color,matrix]); face['dynamic'] = True
        body = cube('body', (0,0,-.07), (.36,.24,.12), rng.choice(body_mats), root); body['dynamic'] = True
        bpy.context.view_layer.update()
        pts = [root.matrix_world @ Vector((x,y,.008)) for x,y in ((-spacing/2,length/2),(spacing/2,length/2),(spacing/2,-length/2),(-spacing/2,-length/2))]
        uv = [project(point) for point in pts]
        if not all(0 <= x < 640 and 0 <= y < 480 for x,y in uv):
            # Keep a visible instance only if its complete keypoint set fits the image.
            for o in [mesh,face,body,root] + [x for x in bpy.data.objects if x.parent == root]:
                if o.name in bpy.data.objects: bpy.data.objects.remove(o,do_unlink=True)
            continue
        bar_corners = [project(root.matrix_world @ Vector((side*spacing/2+dx, y, .008)))
                       for side in (-1,1) for dx in (-.005,.005) for y in (-length/2,length/2)]
        pending.append((cls, color, matrix, large, root, spacing, length, pts, uv, bar_corners))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for cls, color, matrix, large, root, spacing, length, pts, uv, bar_corners in pending:
        paired = sorted(zip(uv, pts), key=lambda pair: pair[0][1])
        paired = sorted(paired[:2], key=lambda pair: pair[0][0]) + sorted(paired[2:], key=lambda pair: pair[0][0], reverse=True)
        ordered = [point for point, _ in paired]
        visibility = []
        for _, point in paired:
            ray = point - cam.location
            hit, location, _, _, obj, _ = s.ray_cast(depsgraph, cam.location, ray.normalized(), distance=ray.length - 0.001)
            visibility.append(1 if hit and obj.parent != root else 2)
        xs, ys = zip(*bar_corners)
        x0, x1 = max(0,min(xs)), min(640,max(xs))
        y0, y1 = max(0,min(ys)), min(480,max(ys))
        labels.append(' '.join(map(str, [cls,(x0+x1)/1280,(y0+y1)/960,(x1-x0)/640,(y1-y0)/480]+[v for (x,y),vis in zip(ordered,visibility) for v in (x/640,y/480,vis)])))
        pos = sum(pts, Vector()) / 4; normal = root.matrix_world.to_quaternion() @ Vector((0,0,1))
        truth.append(dict(class_id=cls, colour=('blue','red')[color], face='matrix' if matrix else 'sticker', large=large, keypoints_camera=[[point.x, -point.y, -point.z] for _, point in paired], visibility=visibility, position=[pos.x,-pos.y,-pos.z], normal=[normal.x,-normal.y,-normal.z], spacing=spacing, length=length))
    stem=f'{index:06d}'
    (a.output/'labels'/f'{stem}.txt').write_text('\n'.join(labels)+'\n' if labels else '')
    (a.output/'truth'/f'{stem}.json').write_text(json.dumps(dict(width=640, height=480, hfov_degrees=69, panels=truth), indent=2) + '\n')
    s.render.filepath=str(a.output/'images'/f'{stem}.jpg')
    bpy.ops.render.render(write_still=True)
