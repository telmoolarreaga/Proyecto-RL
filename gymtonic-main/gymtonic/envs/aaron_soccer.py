# gymtonic/envs/aaron_soccer.py
import logging
import math
import time
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

    def __init__(self, max_speed=1.0, perimeter_side=10.0, goal_target='random', render_mode=None, record_video_file=None):
        super().__init__()
        self.render_mode = render_mode
        self.perimeter_side = float(perimeter_side)
        self.max_speed = float(max_speed)
        self.player_touched_ball = False

        # Decide dirección del gol si pide 'random'
        if goal_target == 'random':
            self.goal_direction = 'right' if np.random.rand() < 0.5 else 'left'
        else:
            self.goal_direction = goal_target  # 'right' o 'left'

        # Conectar con pybullet y crear objetos
        self.connect_to_pybullet(record_video_file)

        # Cargar plano y stadium
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pybullet_plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        pybullet_goal_right_id, pybullet_goal_left_id, pybullet_wall_ids, _ = create_stadium(self.perimeter_side, num_obstacles=0)
        self.pybullet_obstacle_ids = create_fixed_obstacles(field_length=self.perimeter_side, base_height=0.4, cube_size=0.3)

        self.pybullet_goal_right_id = pybullet_goal_right_id
        self.pybullet_goal_left_id = pybullet_goal_left_id
        self.pybullet_wall_ids = pybullet_wall_ids

        # Crear jugador y pelota (posiciones luego reseteadas)
        red_color = [0.8, 0.1, 0.1, 1]
        self.pybullet_player_id = create_player([0, 0, 0.5], red_color)
        self.pybullet_ball_id = create_ball([0, 0, 1.0])

        # Observación: [orientation, speed, player_x, player_y, ball_rel_x, ball_rel_y, goal_rel_x, goal_rel_y]
        low = np.array([-2*math.pi, 0.0, -self.perimeter_side/2, -self.perimeter_side/2, -self.perimeter_side, -self.perimeter_side, -self.perimeter_side, -self.perimeter_side], dtype=np.float32)
        high = np.array([2*math.pi, 10.0, self.perimeter_side/2, self.perimeter_side/2, self.perimeter_side, self.perimeter_side, self.perimeter_side, self.perimeter_side], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # Acción: [rotation_offset_norm (-1..1), forward_force_norm (-1..1)]
        self.action_space = Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        # Variables auxiliares
        self.previous_distance_to_goal = None

    def connect_to_pybullet(self, record_video_file=None):
        # Usar GUI si render_mode == 'human' para que gym.make(..., render_mode='human') funcione
        mode = p.GUI if self.render_mode == 'human' else p.DIRECT
        if record_video_file:
            # opción: grabación si se desea
            self.physics_client = p.connect(mode, options=f"--mp4='{record_video_file}'")
        else:
            self.physics_client = p.connect(mode)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        if self.render_mode == 'human':
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0, -1.5, 0])

    def step_simulation(self):
        p.stepSimulation()
        if self.render_mode == 'human':
            time.sleep(self.SIMULATION_STEP_DELAY)

    def wait_for_simulation(self, sim_steps=120):
        # ejecutar algunos pasos para que la física se estabilice
        self.player_touched_ball = False
        for step in range(sim_steps):
            self.step_simulation()
            if not self.player_touched_ball and self.is_player_touching_ball():
                self.player_touched_ball = True
                # no siempre queremos que el agente lance la pelota automáticamente en wait_for_simulation,
                # pero mantenemos la lógica original (kick_ball) si detecta contacto
                self.kick_ball()
            if step % 10 == 0:
                if self.is_goal() or self.is_ball_out_of_bounds():
                    return

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        # Si goal_direction fue 'random' en init, en cada reset elegimos nuevamente
        if hasattr(self, "goal_direction") and self.goal_direction == 'random':
            self.goal_direction = 'right' if np.random.rand() < 0.5 else 'left'

        # Posicionar agente y pelota aleatoriamente dentro del campo
        rcx = lambda: float(np.random.uniform(-self.perimeter_side/2 + 1, self.perimeter_side/2 - 1))
        rcy = lambda: float(np.random.uniform(-self.perimeter_side/4 + 1, self.perimeter_side/4 - 1))

        quat_starting = p.getQuaternionFromEuler([0, 0, math.pi/2])
        p.resetBasePositionAndOrientation(self.pybullet_player_id, [rcx(), rcy(), 0.5], quat_starting)
        p.resetBaseVelocity(self.pybullet_player_id, [0, 0, 0], [0, 0, 0])

        p.resetBasePositionAndOrientation(self.pybullet_ball_id, [rcx(), rcy(), 1.0], [0, 0, 0, 1])
        p.resetBaseVelocity(self.pybullet_ball_id, [0, 0, 0], [0, 0, 0])

        # Reiniciar métricas
        player_pos, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        goal_pos = np.array([self.perimeter_side/2, 0]) if self.goal_direction == 'right' else np.array([-self.perimeter_side/2, 0])
        self.previous_distance_to_goal = np.linalg.norm(np.array(player_pos[:2]) - goal_pos)

        self.wait_for_simulation()
        obs = self.get_observation()
        return obs, {}

    def execute_action(self, action):
        rotation_offset = float(action[0])
        force = float(action[1])

        position, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        angle = self.get_orientation(self.pybullet_player_id)

        if self.DISCRETE_ROTATION:
            # Mapear rotation_offset [-1,1] -> [-30deg, +30deg]
            max_rot = math.pi / 6.0
            rotation = rotation_offset * max_rot
            # Aplicamos rotación instantánea (más rápido que múltiples pequeños pasos)
            new_angle = angle + rotation
            p.resetBasePositionAndOrientation(self.pybullet_player_id, position, p.getQuaternionFromEuler([0, 0, new_angle]))
            # no hacemos muchos pasos intermedios para no ralentizar
        else:
            # Rotación continua (no usada por defecto)
            new_angle = angle + rotation_offset * 0.1
            p.resetBasePositionAndOrientation(self.pybullet_player_id, position, p.getQuaternionFromEuler([0, 0, new_angle]))

        # Aplicar fuerza en la dirección local X del agente
        self.move_player(force)
        # Ejecutar unos pasos de simulación
        self.wait_for_simulation(sim_steps=40)

    def move_player(self, force):
        # Mapear fuerza desde [-1,1] a un valor usable
        factor = 600.0  # factor reducido para estabilidad
        applied = float(force) * factor * self.max_speed
        local_force = [applied, 0, 0]
        position, orientation = p.getBasePositionAndOrientation(self.pybullet_player_id)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        force_world = rotation_matrix.dot(local_force)
        p.applyExternalForce(self.pybullet_player_id, -1, force_world.tolist(), position, p.WORLD_FRAME)

    def get_oriented_distance_vector(self, object_1_id, object_2_id):
        base_position_1, base_orientation_1 = p.getBasePositionAndOrientation(object_1_id)
        position_2, _ = p.getBasePositionAndOrientation(object_2_id)
        base_rotation_matrix_1 = np.array(p.getMatrixFromQuaternion(base_orientation_1)).reshape(3, 3)
        relative_position = np.array(position_2) - np.array(base_position_1)
        relative_position_base_frame = base_rotation_matrix_1.T.dot(relative_position)
        return relative_position_base_frame

    def get_observation(self):
        my_orientation = self.get_orientation(self.pybullet_player_id)
        velocity, _ = p.getBaseVelocity(self.pybullet_player_id)
        speed = float(math.sqrt(velocity[0]**2 + velocity[1]**2))

        my_pos, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        my_xy = np.array(my_pos[:2], dtype=np.float32)

        # vector orientado al balón (en el sistema del jugador)
        ball_vector = self.get_oriented_distance_vector(self.pybullet_player_id, self.pybullet_ball_id)[:2].astype(np.float32)

        # vector orientado a la línea de gol (usamos el objeto goal id)
        goal_line_id = self.pybullet_goal_right_id if self.goal_direction == 'right' else self.pybullet_goal_left_id
        goal_vector = self.get_oriented_distance_vector(self.pybullet_player_id, goal_line_id)[:2].astype(np.float32)

        obs = np.concatenate(([my_orientation], [speed], my_xy.astype(np.float32), ball_vector, goal_vector)).astype(np.float32)
        return obs

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"
        self.execute_action(action)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Check goal first
        goal = self.is_goal()
        if goal and goal == self.goal_direction:
            reward += 100.0
            terminated = True
            logger.info("Goal scored!")
        elif self.is_ball_out_of_bounds():
            reward -= 5.0
            terminated = True
            logger.info("Ball out of bounds")
        else:
            # Posesión
            if self.player_touched_ball or self.is_player_touching_ball():
                reward += 0.5
                self.player_touched_ball = True

        # Shaping: proximidad al gol (por el jugador)
        player_pos, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        goal_pos = np.array([self.perimeter_side/2, 0]) if self.goal_direction == 'right' else np.array([-self.perimeter_side/2, 0])
        dist_to_goal = np.linalg.norm(np.array(player_pos[:2]) - goal_pos)
        if self.previous_distance_to_goal is not None:
            # si se acerca al gol, recompensa positiva (shaping)
            reward += 0.01 * (self.previous_distance_to_goal - dist_to_goal)
        self.previous_distance_to_goal = dist_to_goal

        # Penalizacion por tiempo para evitar episodios infinitos
        reward -= 0.01

        obs = self.get_observation()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def is_player_touching_ball(self):
        pts = p.getContactPoints(self.pybullet_player_id, self.pybullet_ball_id)
        return len(pts) > 0

    def kick_ball(self):
        agent_position, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        ball_position, _ = p.getBasePositionAndOrientation(self.pybullet_ball_id)
        dir2d = np.array(ball_position[:2]) - np.array(agent_position[:2])
        norm = np.linalg.norm(dir2d)
        if norm < 1e-6:
            return
        kick_direction = np.append(dir2d / norm, 0.0)
        force = 50.0
        p.applyExternalForce(self.pybullet_ball_id, -1, (force * kick_direction).tolist(), ball_position, p.WORLD_FRAME)

    def is_goal(self):
        ball_pos = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0]
        bx, by = ball
