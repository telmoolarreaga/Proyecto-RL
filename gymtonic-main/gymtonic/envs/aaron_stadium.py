# gymtonic/envs/aaron_stadium.py
import math
import os
import numpy as np
import pybullet as p

def create_fixed_obstacles(field_length, base_height=0.4, cube_size=0.5):
    formation_positions = [
        (-2, -1), (-2, 1),
        (2, -1), (2, 1)
    ]
    obstacle_ids = []
    for (x, y) in formation_positions:
        z = base_height + cube_size / 2.0
        pos = [x, y, z]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size/2]*3)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size/2]*3, rgbaColor=[1, 0, 0, 1])
        obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=pos)
        p.changeDynamics(obstacle_id, -1, lateralFriction=1.0, rollingFriction=0.1, spinningFriction=0.1, restitution=0.8)
        obstacle_ids.append(obstacle_id)
    return obstacle_ids

def create_stadium(side_length, num_obstacles=10):
    base_height = 0.4
    _create_turf(side_length, base_height)
    pybullet_wall_ids = _create_perimeter(side_length, thickness=0.1, height=0.4, base_height=base_height, color=[0.7, 0.7, 0.7, 1])
    pybullet_goal_right_id = _create_goal([side_length/2 - 0.10, 0, base_height], p.getQuaternionFromEuler([0, 0, 0]))
    pybullet_goal_left_id = _create_goal([-side_length/2 + 0.10, 0, base_height], p.getQuaternionFromEuler([0, 0, math.pi]))
    obstacle_ids = []
    if num_obstacles > 0:
        # si quieres crear obstáculos dinámicos, implementa create_obstacles
        obstacle_ids = []
    return pybullet_goal_right_id, pybullet_goal_left_id, pybullet_wall_ids, obstacle_ids

def create_player(position=[0,0,0], color=[0.8, 0.1, 0.1, 1]):
    player_id = p.loadURDF("cube.urdf", position, useFixedBase=False, globalScaling=0.4)
    p.changeVisualShape(player_id, -1, rgbaColor=color)
    p.changeDynamics(player_id, -1, restitution=0.8, mass=1.0, lateralFriction=1.0, rollingFriction=1.0, spinningFriction=1.0)
    return player_id

def create_ball(position):
    pybullet_ball_id = p.loadURDF("soccerball.urdf", position, useFixedBase=False, globalScaling=0.3)
    p.changeDynamics(bodyUniqueId=pybullet_ball_id, linkIndex=-1, lateralFriction=1.0, spinningFriction=0.05, rollingFriction=0.05, mass=0.1, restitution=0.8)
    return pybullet_ball_id

def _create_turf(side_length, height=0.4):
    half_size = side_length / 2.0 + 0.8
    half_height = height / 2.0
    color = [0, 0.4, 0, 0.9]
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[half_size, half_size/2, half_height], rgbaColor=color, specularColor=[0,0,0])
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[half_size, half_size/2, half_height])
    turf_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=[0,0,half_height])
    p.changeDynamics(turf_id, -1, lateralFriction=0.5)
    return turf_id

def _create_perimeter(length, thickness, height, base_height, color):
    width = length / 2.0
    half_length = length / 2.0
    half_width = width / 2.0
    half_thickness = thickness / 2.0
    half_height = height / 2.0
    gap_size = 2.0
    segment_length = (width - gap_size) / 2.0
    corner_radius = 1.0

    long_wall_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[half_length - corner_radius, half_thickness, half_height])
    short_wall_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[segment_length / 2.0 - corner_radius/2.0, half_thickness, half_height])
    long_wall_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[half_length-corner_radius, half_thickness, half_height], rgbaColor=[color[0], color[1], color[2], 1])
    short_wall_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[segment_length / 2.0-corner_radius/2.0, half_thickness, half_height], rgbaColor=[color[0], color[1], color[2], 1])

    positions = [
        [0, -half_width + half_thickness, base_height + half_height],
        [0, +half_width - half_thickness, base_height + half_height],
        [half_length - half_thickness, half_width - segment_length / 2.0 - corner_radius/2.0, base_height + half_height],
        [half_length - half_thickness, -half_width + segment_length / 2.0 + corner_radius/2.0, base_height + half_height],
        [-half_length + half_thickness, half_width - segment_length / 2.0 - corner_radius/2.0, base_height + half_height],
        [-half_length + half_thickness, -half_width + segment_length / 2.0 + corner_radius/2.0, base_height + half_height]
    ]
    orientations = [p.getQuaternionFromEuler([0,0,0])] * 2 + [p.getQuaternionFromEuler([0,0,1.5708])] * 4

    pybullet_wall_ids = []
    for i in range(2):
        id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=long_wall_shape, baseVisualShapeIndex=long_wall_visual, basePosition=positions[i], baseOrientation=orientations[i])
        p.changeDynamics(id, -1, restitution=0.8)
        pybullet_wall_ids.append(id)
    for i in range(2, 6):
        id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=short_wall_shape, baseVisualShapeIndex=short_wall_visual, basePosition=positions[i], baseOrientation=orientations[i])
        p.changeDynamics(id, -1, restitution=0.8)
        pybullet_wall_ids.append(id)

    # Cantos curvos (opcional)
    pybullet_corners_1 = _create_curved_corner([half_length - half_thickness - corner_radius, half_width - half_thickness - corner_radius, base_height + half_height], 2*half_height, radius=1, orientation=0, color=color)
    pybullet_corners_2 = _create_curved_corner([half_length - half_thickness - corner_radius, -half_width + half_thickness + corner_radius, base_height + half_height], 2*half_height, radius=1, orientation=-math.pi/2, color=color)
    pybullet_corners_3 = _create_curved_corner([-half_length + half_thickness + corner_radius, half_width - half_thickness - corner_radius, base_height + half_height], 2*half_height, radius=1, orientation=math.pi/2, color=color)
    pybullet_corners_4 = _create_curved_corner([-half_length + half_thickness + corner_radius, -half_width + half_thickness + corner_radius, base_height + half_height], 2*half_height, radius=1, orientation=math.pi, color=color)

    pybullet_wall_ids.extend(pybullet_corners_1)
    pybullet_wall_ids.extend(pybullet_corners_2)
    pybullet_wall_ids.extend(pybullet_corners_3)
    pybullet_wall_ids.extend(pybullet_corners_4)

    return pybullet_wall_ids

def _create_curved_corner(position, height, radius=1, orientation=0, color=[0,0,0,1], num_segments=12):
    angle_step = np.pi / 2.0 / num_segments
    pybullet_object_ids = []
    for i in range(num_segments):
        angle = i * angle_step
        x = position[0] + radius * np.cos(angle + orientation)
        y = position[1] + radius * np.sin(angle + orientation)
        z = position[2]
        cylinder_collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=height)
        cylinder_id = p.createMultiBody(baseCollisionShapeIndex=cylinder_collision_id, basePosition=[x,y,z], baseOrientation=p.getQuaternionFromEuler([0,0,angle + orientation + np.pi/2]))
        p.changeVisualShape(cylinder_id, -1, rgbaColor=color)
        p.changeDynamics(cylinder_id, -1, restitution=0.8)
        pybullet_object_ids.append(cylinder_id)
    return pybullet_object_ids

def _create_goal(position, orientation):
    script_dir = os.path.dirname(__file__)
    goal_path = os.path.join(script_dir, "meshes", "goal.obj")
    if not os.path.exists(goal_path):
        # Si no existe el mesh, crear un simple placeholder box
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 0.3])
        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 0.3], rgbaColor=[1,1,1,1])
        goal_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=orientation)
    else:
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=goal_path, meshScale=[1,1,1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=goal_path, meshScale=[1,1,1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        goal_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=orientation)
    p.changeDynamics(goal_id, -1, restitution=0.8)
    return goal_id
