# gymtonic/envs/aaron_soccer.py
import logging
import math
import time
from turtle import pos
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from typing import Any
from .aaron_stadium import create_stadium, create_player, create_ball, create_fixed_obstacles

logger = logging.getLogger(__name__)

class SoccerSingleEnv(Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    SIMULATION_STEP_DELAY = 1.0 / 240.0
    DISCRETE_ROTATION = True

    def __init__(self, max_speed=1.0, perimeter_side=10.0, goal_target='left', render_mode=None, record_video_file=None):
        super().__init__()
        self.render_mode = render_mode
        self.perimeter_side = float(perimeter_side)
        self.max_speed = float(max_speed)
        self.player_touched_ball = False

        # Decides goal target if random is selected
        '''
        if goal_target == 'random':
            self.goal_direction = 'right' if np.random.rand() < 0.5 else 'left'
        else:
            self.goal_direction = goal_target  # 'right' o 'left'
        '''
        self.goal_target = goal_target
        

        # Connect to PyBullet and create objects
        self.connect_to_pybullet(record_video_file)

        # Load plane and stadium
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pybullet_plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        pybullet_goal_right_id, pybullet_goal_left_id, pybullet_wall_ids, = create_stadium(self.perimeter_side)
        self.pybullet_obstacle_ids = create_fixed_obstacles(field_length=self.perimeter_side, base_height=0.4, cube_size=0.3)

        self.pybullet_goal_right_id = pybullet_goal_right_id
        self.pybullet_goal_left_id = pybullet_goal_left_id
        self.pybullet_wall_ids = pybullet_wall_ids

        # Create player and ball
        red_color = [0.8, 0.1, 0.1, 1]
        self.pybullet_player_id = create_player([0, 0, 0.5], red_color)
        self.pybullet_ball_id = create_ball([0, 0, 1.0])

        # Create moving goalie
        self.pybullet_goalie_id = None
        self.goalie_direction = 1  # 1 = right, -1 = left


        # Observation: [orientation, speed, player_x, player_y, ball_rel_x, ball_rel_y, goal_rel_x, goal_rel_y]
        low = np.array([-2*math.pi, 0.0, -self.perimeter_side/2, -self.perimeter_side/2, -self.perimeter_side, -self.perimeter_side, -self.perimeter_side, -self.perimeter_side], dtype=np.float32)
        high = np.array([2*math.pi, 10.0, self.perimeter_side/2, self.perimeter_side/2, self.perimeter_side, self.perimeter_side, self.perimeter_side, self.perimeter_side], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # Action: [rotation_offset_norm (-1..1), forward_force_norm (-1..1)]
        self.action_space = Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        # Auxiliary variables
        self.previous_distance_to_goal = None

    # Pybullet connection
    def connect_to_pybullet(self, record_video_file=None):
        if record_video_file is not None:
            self.physicsClient = p.connect(p.GUI if self.render_mode else p.DIRECT, options=f"--mp4='{record_video_file}'")
        else:
            self.physicsClient = p.connect(p.GUI if self.render_mode else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        if self.render_mode == 'human':
            # Hide debug panels in PyBullet
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

            # Reorient the debug camera
            p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0,-1.5,0])

    # Create moving goalie
    def create_moving_goalie(self):
        # Initial position in the center of the goal
        if self.goal_target == 'right':
         x = self.perimeter_side / 2 - 0.1
        else:
         x = -self.perimeter_side / 2 + 0.1
        y = 0  # center
        z = 0.5  # height
        color = [0, 0, 1, 1]  # blue
        size = 0.3
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2, size/2, size/2])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2, size/2, size/2], rgbaColor=color)
        goalie_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                  baseVisualShapeIndex=visual, basePosition=[x, y, z])
        return goalie_id
    
    # Simulation Control
    def step_simulation(self):
        p.stepSimulation()
        self.move_goalie()  # move goalie each step
        if self.render_mode == 'human':
            time.sleep(self.SIMULATION_STEP_DELAY)

    def wait_for_simulation(self, sim_steps=500):
        self.player_touched_ball = False
        for step in range(sim_steps):
            self.step_simulation()
            if not self.player_touched_ball and self.is_player_touching_ball():
                self.player_touched_ball = True
                self.kick_ball()
            # Every 10 sim steps, check status
            if step%10 == 0:
                if self.is_goal() or self.is_ball_out_of_bounds():
                    return
                # If the player is not moving, return
                player_velocity, _ = p.getBaseVelocity(self.pybullet_player_id)
                if np.linalg.norm(player_velocity) < 2*self.max_speed:
                    return

    # Gym Api
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        if not p.isConnected():
            self.connnect_to_pybullet()

        goal_color = [0.5, 0.9, 0.6, 1]

        if self.goal_target == 'random':
            self.goal_direction = 'right' if np.random.rand() > 0.5 else 'left'
        else:
            self.goal_direction = self.goal_target

        if self.goal_direction == 'right':
            p.changeVisualShape(self.pybullet_goal_right_id, -1, rgbaColor=goal_color)
            p.changeVisualShape(self.pybullet_goal_left_id, -1, rgbaColor=[1, 1, 1, 1])
        else:
            p.changeVisualShape(self.pybullet_goal_left_id, -1, rgbaColor=goal_color)
            p.changeVisualShape(self.pybullet_goal_right_id, -1, rgbaColor=[1, 1, 1, 1])
        # Create goalie if not existing
        if self.pybullet_goalie_id is None:
            self.pybullet_goalie_id = self.create_moving_goalie()


        limit_spawn_perimeter_x = self.perimeter_side / 2 -1
        limit_spawn_perimeter_y = self.perimeter_side / 4 -1

        random_coor_x = lambda: np.random.uniform(-limit_spawn_perimeter_x, limit_spawn_perimeter_x)
        #random_coor_x = lambda: np.random.uniform(0, limit_spawn_perimeter_x)
        random_coor_y = lambda: np.random.uniform(-limit_spawn_perimeter_y, limit_spawn_perimeter_y)

        # Player facing upwards
        quat_starting = p.getQuaternionFromEuler([0, 0, math.pi/2])

        p.resetBasePositionAndOrientation(self.pybullet_player_id, [random_coor_x(), random_coor_y(),  0.5], quat_starting)
        p.resetBaseVelocity(self.pybullet_player_id, [0, 0, 0], [0, 0, 0])

        # Random ball position version
        p.resetBasePositionAndOrientation(self.pybullet_ball_id, [random_coor_x(), random_coor_y(),  1], [0, 0, 0, 1])
        p.resetBaseVelocity(self.pybullet_ball_id, [0, 0, 0], [0, 0, 0])
        self.wait_for_simulation()

        info = {}
        obs = self.get_observation()
        return obs, info

    def execute_action(self, action):

        rotation_offset = action[0]
        force = action[1]

        # Rotate the player towards the force direction
        position, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)

        #linear_velocity, angular_velocity = p.getBaseVelocity(pybullet_object_id) # Conserving inertia for the next step
        linear_velocity = angular_velocity = [0,0,0] # Starting still in next step

        angle = self.get_orientation(self.pybullet_player_id)
        if self.DISCRETE_ROTATION:
            rotation_offset = rotation_offset * math.pi / 6 # 30 degrees max rotation
            #angle += rotation_offset
            n_steps_rotation = abs(int(rotation_offset / (math.pi/360*2)))
            for i in range(n_steps_rotation):
                angle += rotation_offset/n_steps_rotation
                p.resetBasePositionAndOrientation(self.pybullet_player_id, position, p.getQuaternionFromEuler([0, 0, angle]))
                self.step_simulation()
            p.resetBaseVelocity(self.pybullet_player_id, linearVelocity=linear_velocity, angularVelocity=angular_velocity)
        """
        else:
            # Necessary to reset to avoid weird orientations after colission with ball or wall
            p.resetBasePositionAndOrientation(self.pybullet_player_id, position, p.getQuaternionFromEuler([0, 0, angle]))
            p.applyExternalTorque(self.pybullet_player_id, -1, [0,0,rotation_offset*100], p.WORLD_FRAME)
            self.wait_until_stable()
        """

        self.move_player(force)        
        self.wait_for_simulation()

    def move_player(self, force):
        factor = 1000
        force *= factor * self.max_speed
        force = [force, 0, 0]
        position, orientation = p.getBasePositionAndOrientation(self.pybullet_player_id)
        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # Transform the force vector from local to world coordinates
        force_world = np.dot(rotation_matrix, force)

        p.applyExternalForce(self.pybullet_player_id, -1, force_world, position, p.WORLD_FRAME)

    def get_oriented_distance_vector(self, object_1_id, object_2_id):
        # Get the base position and orientation of object_1
        base_position_1, base_orientation_1 = p.getBasePositionAndOrientation(object_1_id)

        # Get the base position of object_2
        position_2, _ = p.getBasePositionAndOrientation(object_2_id)

        # Convert the position of object_2 to the base frame of object_1
        # Compute the inverse of the base's world orientation
        base_rotation_matrix_1 = np.array(p.getMatrixFromQuaternion(base_orientation_1)).reshape(3, 3)

        # Compute the relative position of object_2 in the world frame
        relative_position = np.array(position_2) - np.array(base_position_1)

        # Transform the relative position to the base frame of object_1
        relative_position_base_frame = np.dot(base_rotation_matrix_1.T, relative_position)
        
        return relative_position_base_frame
    
    def move_goalie(self):
        if self.pybullet_goalie_id is None:
            return

        pos, ori = p.getBasePositionAndOrientation(self.pybullet_goalie_id)
    
        # low pace
        y = pos[1] + self.goalie_direction * 0.005  

        # exact limits of the goal
        goal_width = self.perimeter_side * 0.22  
        half_goal_width = goal_width / 2

        if y > half_goal_width or y < -half_goal_width:
            self.goalie_direction *= -1
            y = max(min(y, half_goal_width), -half_goal_width)

        # fixed X position on the goal line
        if self.goal_target == 'right':
            x = self.perimeter_side / 2 - 0.1
        else:
            x = -self.perimeter_side / 2 + 0.1
        # maintain Z
        z = pos[2]

        p.resetBasePositionAndOrientation(self.pybullet_goalie_id, [x, y, z], ori)

    # Observations
    def get_observation(self):
        obs = np.array([], dtype=np.float32)

        my_orientation = self.get_orientation(self.pybullet_player_id)
        obs = np.concatenate((obs, [my_orientation]), dtype=np.float32)

        # Add speed
        velocity, _ = p.getBaseVelocity(self.pybullet_player_id)
        velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        obs = np.concatenate((obs, [velocity]), dtype=np.float32)

        my_pos, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        obs = np.concatenate((obs, my_pos[:2]), dtype=np.float32)

        # Now the ball
        ball_pos, _ = p.getBasePositionAndOrientation(self.pybullet_ball_id)
        ball_vector = self.get_oriented_distance_vector(self.pybullet_player_id, self.pybullet_ball_id)[:2]
        obs = np.concatenate((obs, ball_vector), dtype=np.float32)
        #obs = np.concatenate((obs, ball_pos[:2]), dtype=np.float32)

        goal_line_id = self.pybullet_goal_right_id if self.goal_direction == 'right' else self.pybullet_goal_left_id

        goal_line_vector = self.get_oriented_distance_vector(self.pybullet_player_id, goal_line_id)[:2]
        obs = np.concatenate((obs, goal_line_vector), dtype=np.float32)

        return obs

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        self.execute_action(action)

        truncated = False
        reward = 0
        info = {}  
        terminated = False

        goal = self.is_goal()
    
        # Reward por gol
        if goal == self.goal_direction:
            reward += 100
            terminated = True
            logger.info(f"Goal scored!")
        elif self.is_ball_out_of_bounds():  # after goal
            terminated = True
            logger.info("Ball is out of bounds")
        else:
            # ball posession 
            if self.player_touched_ball:
                reward += 1  # small reward for touching the ball
                logger.info(f"Player touched the ball")

            #  Penalty for time
            reward -= 0.1

            #  Penalty for touching obstacles
            for obs_id in self.pybullet_obstacle_ids:
                if p.getContactPoints(self.pybullet_player_id, obs_id):
                    reward -= 0.5
                    logger.info(f"Player touched an obstacle!")

        obs = self.get_observation()
        return obs, reward, terminated, truncated, info
    
    def is_player_touching_ball(self):
        if p.getContactPoints(self.pybullet_player_id, self.pybullet_ball_id):
            return True
        return False

    def kick_ball(self):
        # Get the position of the agent and the ball
        agent_position,_ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        ball_position,_ = p.getBasePositionAndOrientation(self.pybullet_ball_id)

        # Calculate the direction of the kick
        kick_direction = np.array(ball_position[:2]) - np.array(agent_position[:2])
        kick_direction = kick_direction / np.linalg.norm(kick_direction)
        kick_direction = np.append(kick_direction, 0)  # Add a zero z-component

        # Apply the kick
        force = 50
        p.applyExternalForce(self.pybullet_ball_id, -1, force * kick_direction, ball_position, p.WORLD_FRAME)

    def is_goal(self):
        """
        Checks if the ball has completely crossed any of the goal gaps.

        Returns:
            None if no goal,
            'right' if the goal was scored in the right hand side goal gap
            'left' if the goal was scored in the left hand side goal gap,
        """
        # Get the position and radius of the ball
        ball_position = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0]
        ball_x = ball_position[0]
        ball_y = ball_position[1]
        ball_radius = 0.2  # Assume this ball_radius
        
        # Define the dimensions based on the pitch creation
        length = self.perimeter_side
        width = self.perimeter_side / 2
        thickness = 0.1
        gap_size = width * 0.25
        segment_length = (width - gap_size) / 2

        # Check if the ball is completely within the right goal gap
        right_goal_x = length / 2
        if (ball_x - ball_radius > right_goal_x - thickness and
            -segment_length / 2 < ball_y < segment_length / 2):
            return 'right'
        
        # Check if the ball is completely within the left goal gap
        left_goal_x = -length / 2
        if (ball_x + ball_radius < left_goal_x + thickness and
            -segment_length / 2 < ball_y < segment_length / 2):
            return 'left'

        # No goal detected
        return False

    def _is_object_out_of_bounds(self, object_id):  
        margin = 0.5
        pitch_length = self.perimeter_side / 2 + margin
        pitch_width = self.perimeter_side / 4 + margin
        object_pos,_ = p.getBasePositionAndOrientation(object_id)
        object_pos = np.array(object_pos[:2])
        if object_pos[0] < -pitch_length or object_pos[0] > pitch_length or object_pos[1] < -pitch_width or object_pos[1] > pitch_width:
            return True
        return False

    def is_player_out_of_bounds(self):
        if self._is_object_out_of_bounds(self.pybullet_player_id):
            return True
        return False

    def is_ball_out_of_bounds(self):
        return self._is_object_out_of_bounds(self.pybullet_ball_id)

    def get_object_distances(self, this_object_id, object_ids):
        distances = []
        pos1, _ = p.getBasePositionAndOrientation(this_object_id)
        for obj_id in object_ids:
            if obj_id != this_object_id:  # Avoid comparing the object to itself
                pos2, _ = p.getBasePositionAndOrientation(obj_id)
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                distances.append((obj_id, pos2, distance))
        
        # Sort distances based on the second element (distance)
        distances.sort(key=lambda x: x[2])
        
        return distances
    
    def get_orientation(self, pybullet_object_id):
        # Get the orientation of the object
        _, orientation = p.getBasePositionAndOrientation(pybullet_object_id)
        
        # Convert quaternion to euler angles
        euler_angles = p.getEulerFromQuaternion(orientation)
        
        # Extract the angle around the z-axis
        # euler_angles returns (roll, pitch, yaw)
        # yaw is the angle around the z-axis
        angle_z = euler_angles[2]  # radians
        
        return angle_z

    # Utilities
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == 'human':
            p.stepSimulation()
        return None
    
    def close(self):
        p.disconnect()