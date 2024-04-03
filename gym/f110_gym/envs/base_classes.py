# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Prototype of base classes
Replacement of the old RaceCar, Simulator classes in C++
Author: Hongrui Zheng
"""

import numpy as np
from numba import njit
from scipy import integrate

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, vehicle_dynamics_mb, init_mb, pid, accl_constraints, steering_constraint
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110_gym.envs.collision_models import get_vertices, collision_multiple
DO_SCAN = True


class RaceCar(object):
    """
    Base level race car class, handles the physics and laser scan of a single vehicle

    Data Members:
        params (dict): vehicle parameters dictionary
        is_ego (bool): ego identifier
        time_step (float): physics timestep
        num_beams (int): number of beams in laser
        fov (float): field of view of laser
        state (for dynamic_ST model) (np.ndarray (7, )): state vector [x, y, theta, vel, steer_angle, ang_vel, slip_angle]
        state (for MB model) (np.ndarray (29, )): state vector [x, y, steering angle of front wheels, velocity in x-direction,
            yaw angle, yaw rate, roll angle, roll rate, pitch angle, pitch rate, velocity in y-direction, z-position, velocity in z-direction,
            roll angle front, roll rate front, velocity in y-direction front, z-position front, velocity in z-direction front, roll angle rear,
            roll rate rear, velocity in y-direction rear, z-position rear, velocity in z-direction rear, left front wheel angular speed,
            right front wheel angular speed, left rear wheel angular speed, right rear wheel angular speed, delta_y_f, delta_y_r]
        odom (np.ndarray(13, )): odometry vector [x, y, z, qx, qy, qz, qw, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        accel (float): current acceleration input
        steer_angle_vel (float): current steering velocity input
        in_collision (bool): collision indicator

    """

    # static objects that don't need to be stored in class instances
    scan_simulator = None
    cosines = None
    scan_angles = None
    side_distances = None

    def __init__(self, model, steering_control_mode, drive_control_mode, params, seed, is_ego=False, time_step=0.01,
                 num_beams=1080, fov=4.7):
        """
        Init function

        Args:
            model (str): vehicle model to use. Options: 'dynamic_ST' - dynamic single track model, 'MB' - multi body model
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max', 'length', 'width'}
            is_ego (bool, default=False): ego identifier
            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser

        Returns:
            None
        """

        # initialization
        self.model = model
        self.drive_control_mode = drive_control_mode
        self.steering_control_mode = steering_control_mode
        self.params = params
        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov
        self.tire_forces = np.zeros(8)
        self.longitudinal_slip = np.zeros(4)
        self.lateral_slip = np.zeros(4)
        self.vertical_tire_forces = np.zeros(4)

        if self.model == 'dynamic_ST':
            # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
            self.state = np.zeros((7,))
        elif self.model == 'MB':
            params_array = np.array(list(self.params.values()))
            self.state = init_mb(np.zeros((7,)), params_array)

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # steering delay buffer
        self.steer_buffer = np.empty((0,))
        self.steer_buffer_size = 2

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if RaceCar.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            RaceCar.scan_simulator = ScanSimulator2D(num_beams, fov)

            scan_ang_incr = RaceCar.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            RaceCar.cosines = np.zeros((num_beams,))
            RaceCar.scan_angles = np.zeros((num_beams,))
            RaceCar.side_distances = np.zeros((num_beams,))

            dist_sides = params['width'] / 2.
            dist_fr = (params['lf'] + params['lr']) / 2.

            for i in range(num_beams):
                angle = -fov / 2. + i * scan_ang_incr
                RaceCar.scan_angles[i] = angle
                RaceCar.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi / 2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi / 2.)
                        to_fr = dist_fr / np.sin(angle - np.pi / 2.)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi / 2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi / 2)
                        to_fr = dist_fr / np.sin(-angle - np.pi / 2)
                        RaceCar.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """
        Updates the physical parameters of the vehicle
        Note that does not need to be called at initialization of class anymore

        Args:
            params (dict): new parameters for the vehicle

        Returns:
            None
        """
        self.params = params

    def set_map(self, map_path, map_ext):
        """
        Sets the map for scan simulator
        
        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file
        """
        RaceCar.scan_simulator.set_map(map_path, map_ext)

    def reset(self, pose, steering_angle=0, velocity=0, yaw_rate=0, beta=0):
        """
        Resets the vehicle to a pose
        
        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to

        Returns:
            None
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # clear state
        if self.model == 'dynamic_ST':
            # print('pose', pose)
            # self.state = pose
            self.state = np.zeros((7,))
            self.state[0:2] = pose[0:2]
            self.state[2] = steering_angle
            self.state[3] = velocity
            self.state[4] = pose[2]
            self.state[5] = yaw_rate
            self.state[6] = beta
        elif self.model == 'MB':
            params_array = np.array(list(self.params.values()))
            
            if len(pose) == 29:
                self.state = pose
            else:
                self.state = init_mb(np.array([pose[0], pose[1],
                                            steering_angle, velocity,
                                            pose[2], yaw_rate,
                                            beta]), params_array)
        self.steer_buffer = np.empty((0,))
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)

    def ray_cast_agents(self, scan):
        """
        Ray cast onto other agents in the env, modify original scan

        Args:
            scan (np.ndarray, (n, )): original scan range array

        Returns:
            new_scan (np.ndarray, (n, )): modified scan
        """

        # starting from original scan
        new_scan = scan

        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get vertices of current oppoenent
            opp_vertices = get_vertices(opp_pose, self.params['length'], self.params['width'])

            new_scan = ray_cast(np.append(self.state[0:2], self.state[4]), new_scan, self.scan_angles, opp_vertices)

        return new_scan

    def check_ttc(self, current_scan):
        """
        Check iTTC against the environment, sets vehicle states accordingly if collision occurs.
        Note that this does NOT check collision with other agents.

        state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        Args:
            current_scan

        Returns:
            None
        """

        in_collision = check_ttc_jit(current_scan, self.state[3], self.scan_angles, self.cosines, self.side_distances, self.ttc_thresh)

        # if in collision stop vehicle
        if in_collision:
            self.state[3:] = 0.
            self.accel = 0.0
            self.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision

    @staticmethod
    def func_MB(t, x, u, p, use_kinematic):
        f, F_x_LF, F_x_RF, F_x_LR, F_x_RR, F_y_LF, F_y_RF, F_y_LR, F_y_RR, \
            s_lf, s_rf, s_lr, s_rr, alpha_LF, alpha_RF, alpha_LR, alpha_RR, \
            F_z_LF, F_z_RF, F_z_LR, F_z_RR = vehicle_dynamics_mb(x, u, p, use_kinematic)
        return f

    @staticmethod
    def func_MB2(x, t, u, p, use_kinematic):
        f, F_x_LF, F_x_RF, F_x_LR, F_x_RR, F_y_LF, F_y_RF, F_y_LR, F_y_RR, \
            s_lf, s_rf, s_lr, s_rr, alpha_LF, alpha_RF, alpha_LR, alpha_RR, \
            F_z_LF, F_z_RF, F_z_LR, F_z_RR = vehicle_dynamics_mb(x, u, p, use_kinematic)
        return f

    def update_pose(self, raw_steer, drive):
        """
        Steps the vehicle's physical simulation

        Args:
            steer (float): desired steering angle
            drive (float): desired velocity/acceleration

        Returns:
            current_scan
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # steering delay
        # steer = 0.
        # if self.steer_buffer.shape[0] < self.steer_buffer_size:
        #     steer = 0.
        #     self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        # else:
        #     steer = self.steer_buffer[-1]
        #     self.steer_buffer = self.steer_buffer[:-1]
        #     self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        steer = raw_steer

        if self.steering_control_mode != 'vel' or self.drive_control_mode != 'acc':
            # steering angle velocity input to steering velocity acceleration input
            accl, sv = pid(drive, steer, self.state[3], self.state[2], self.params['sv_max'], self.params['a_max'],
                        self.params['v_max'], self.params['v_min'])
        
        if self.drive_control_mode == 'acc':
            if drive > self.params['a_max']:
                accl = self.params['a_max']
            elif drive < -self.params['a_max']:
                accl = -self.params['a_max']
            else:
                accl = drive

        if self.steering_control_mode == 'vel':
            sv = steer
        integration_method = 'LSODA_old'  # 'LSODA'  'euler' 'LSODA_old' 'RK45'

        # update physics, get RHS of diff'eq
        if self.model == 'dynamic_ST':
            f = vehicle_dynamics_st(
                self.state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])
            # update state
            self.state = self.state + f * self.time_step
        elif self.model == 'MB':
            params_array = np.array(list(self.params.values()))

            # steering constraints
            s_min = params_array[2]  # minimum steering angle [rad]
            s_max = params_array[3]  # maximum steering angle [rad]
            sv_min = params_array[4]  # minimum steering velocity [rad/s]
            sv_max = params_array[5]  # maximum steering velocity [rad/s]

            # longitudinal constraints
            v_min = params_array[6]  # minimum velocity [m/s]
            v_max = params_array[7]  # minimum velocity [m/s]
            v_switch = params_array[8]  # switching velocity [m/s]
            a_max = params_array[9]  # maximum absolute acceleration [m/s^2]

            # split of brake and engine torque
            T_sb = params_array[40]
            T_se = params_array[41]

            m = params_array[10]  # vehicle mass [kg]  MASS

            R_w = params_array[39]  # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

            u = np.array([steering_constraint(self.state[2], sv, s_min, s_max, sv_min, sv_max),
                          accl_constraints(self.state[3], accl, v_switch, a_max, v_min, v_max)])

            if u[1] > 0:
                T_B = 0.0
                T_E = m * R_w * u[1]
            else:
                T_B = m * R_w * u[1]
                T_E = 0.0

            front_tire_force = 0.5 * T_sb * T_B + 0.5 * T_se * T_E
            rear_tire_force = 0.5 * (1 - T_sb) * T_B + 0.5 * (1 - T_se) * T_E

            control_input = np.array([u[0], front_tire_force, rear_tire_force])

            if np.abs(self.state[3]) < 0.01:
                use_kinematic = True
            else:
                use_kinematic = False

            if integration_method == 'euler':
                f = vehicle_dynamics_mb(self.state, control_input, params_array, use_kinematic)
                # update state
                self.state = self.state + f * self.time_step
            elif integration_method == 'LSODA':
                x_left = integrate.solve_ivp(self.func_MB, (0.0, self.time_step),
                                             self.state, method='LSODA',
                                             args=(control_input, params_array, use_kinematic))
                self.state = x_left.y[:, -1]
            elif integration_method == 'RK45':
                x_left = integrate.solve_ivp(self.func_MB, (0.0, self.time_step),
                                             self.state, method='RK45',
                                             args=(control_input, params_array, use_kinematic))
                self.state = x_left.y[:, -1]
            elif integration_method == 'LSODA_old':

                x_left = integrate.odeint(self.func_MB2, self.state,
                                          np.array([0.0, self.time_step]),
                                          args=(control_input, params_array, use_kinematic),
                                          mxstep=10000, full_output=1)
                _, F_x_LF, F_x_RF, F_x_LR, F_x_RR, F_y_LF, F_y_RF, F_y_LR, F_y_RR, \
                    s_lf, s_rf, s_lr, s_rr, alpha_LF, alpha_RF, alpha_LR, alpha_RR, \
                    F_z_LF, F_z_RF, F_z_LR, F_z_RR = vehicle_dynamics_mb(self.state, control_input, params_array, use_kinematic)
                self.tire_forces = np.array([F_x_LF, F_x_RF, F_x_LR, F_x_RR, F_y_LF, F_y_RF, F_y_LR, F_y_RR])
                self.longitudinal_slip = np.array([s_lf, s_rf, s_lr, s_rr])
                self.lateral_slip = np.array([alpha_LF, alpha_RF, alpha_LR, alpha_RR])
                self.vertical_tire_forces = np.array([F_z_LF, F_z_RF, F_z_LR, F_z_RR])
                self.state = x_left[0][1]

            for iState in range(23, 27):
                if self.state[iState] < 0.0:
                    self.state[iState] = 0.0

        # bound yaw angle
        if self.state[4] > 2 * np.pi:
            self.state[4] = self.state[4] - 2 * np.pi
        elif self.state[4] < 0:
            self.state[4] = self.state[4] + 2 * np.pi

        # update scan
        current_scan = self.get_current_scan()

        return current_scan

    def get_current_scan(self):
        return RaceCar.scan_simulator.scan(np.append(self.state[0:2], self.state[4]), self.scan_rng)

    def update_opp_poses(self, opp_poses):
        """
        Updates the vehicle's information on other vehicles

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents

        Returns:
            None
        """
        self.opp_poses = opp_poses

    def update_scan(self, agent_scans, agent_index):
        """
        Steps the vehicle's laser scan simulation
        Separated from update_pose because needs to update scan based on NEW poses of agents in the environment

        Args:
            agent scans list (modified in-place),
            agent index (int)

        Returns:
            None
        """

        current_scan = agent_scans[agent_index]

        # check ttc
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)

        agent_scans[agent_index] = new_scan


class Simulator(object):
    """
    Simulator class, handles the interaction and update of all vehicles in the environment

    Data Members:
        num_agents (int): number of agents in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents
        agents (list[RaceCar]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agent
        collision_idx (np.ndarray(num_agents, )): which agent is each agent in collision with

    """

    def __init__(self, model, steering_control_mode, drive_control_mode, params, num_agents, seed, time_step=0.01,
                 ego_idx=0, scan_beam=1080):
        """
        Init function

        Args:
            model (str): vehicle model to use. Options: 'dynamic_ST' - dynamic single track model, 'MB' - multi body model
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            num_agents (int): number of agents in the environment
            seed (int): seed of the rng in scan simulation
            time_step (float, default=0.01): physics time step
            ego_idx (int, default=0): ego vehicle's index in list of agents

        Returns:
            None
        """
        self.model = model
        self.drive_control_mode = drive_control_mode
        self.steering_control_mode = steering_control_mode
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))

        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(self.model, self.steering_control_mode, self.drive_control_mode, params, self.seed,
                                  time_step=self.time_step, is_ego=True, num_beams=scan_beam)
                self.agents.append(ego_car)
            else:
                agent = RaceCar(self.model, self.steering_control_mode, self.drive_control_mode, params, self.seed,
                                time_step=self.time_step, is_ego=False, num_beams=scan_beam)
                self.agents.append(agent)
                
        # self.observations = self.get_observations(np.zeros((self.num_agents, 2)))
        self.observations = None

    def set_map(self, map_path, map_ext):
        """
        Sets the map of the environment and sets the map for scan simulator of each agent

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)

    def update_params(self, params, agent_idx=-1):
        """
        Updates the params of agents, if an index of an agent is given, update only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update, if negative, update all agents

        Returns:
            None
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError('Index given is out of bounds for list of agents.')

    def check_collision(self):
        """
        Checks for collision between agents using GJK and agents' body vertices

        Args:
            None

        Returns:
            None
        """
        # get vertices of all agents
        all_vertices = np.empty((self.num_agents, 4, 2))
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2], self.agents[i].state[4]), self.params['length'], self.params['width'])
        self.collisions, self.collision_idx = collision_multiple(all_vertices)

    def step(self, control_inputs):
        """
        Steps the simulation environment

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents, first column is desired steering angle, second column is desired velocity
        
        Returns:
            observations (dict): dictionary for observations: poses of agents, current laser scan of each agent, collision indicators, etc.
        """

        # doing step for all agents
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])

        observations = self.get_observations(control_inputs)
        self.observations = observations

        return observations

    def get_observations(self, control_inputs):
        # get_current_scan

        agent_scans = []

        # looping over agents
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            if DO_SCAN:
                current_scan = agent.get_current_scan()
                agent_scans.append(current_scan)

            # update sim's information of agent poses
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])

        # check collisions between all agents
        self.check_collision()

        for i, agent in enumerate(self.agents):
            if DO_SCAN:
                # update agent's information on other agents
                opp_poses = np.concatenate((self.agent_poses[0:i, :], self.agent_poses[i + 1:, :]), axis=0)
                agent.update_opp_poses(opp_poses)

                # update each agent's current scan based on other agents
                agent.update_scan(agent_scans, i)

            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.

        # fill in observations
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # collision_angles is removed from observations

        # observations = {'ego_idx': self.ego_idx,
        #                 'scans': [],
        #                 'poses_x': [],
        #                 'poses_y': [],
        #                 'poses_theta': [],
        #                 'linear_vels_x': [],
        #                 'linear_vels_y': [],
        #                 'ang_vels_z': [],
        #                 'collisions': self.collisions}
        # for i, agent in enumerate(self.agents):
        #     observations['scans'].append(agent_scans[i])
        #     observations['poses_x'].append(agent.state[0])
        #     observations['poses_y'].append(agent.state[1])
        #     observations['poses_theta'].append(agent.state[4])
        #     observations['linear_vels_x'].append(agent.state[3])
        #     observations['linear_vels_y'].append(0.)
        #     observations['ang_vels_z'].append(agent.state[5])
        observations = {'ego_idx': self.ego_idx,
                        'scans': [],
                        'poses_x': [],
                        'poses_y': [],
                        'poses_theta': [],
                        'linear_vels_x': [],
                        'linear_vels_y': [],
                        'ang_vels_z': [],
                        'collisions': self.collisions,
                        'x1': [], # x-pos
                        'x2': [], # y-pos
                        'x3': [], # steer
                        'x4': [], # vx
                        'x5': [], # yaw angle
                        'x6': [], # yaw rate
                        # 'x11': [], # vy
                        # 'x12': [], # height
                        # 'x13': [], # vz
                        # 'x16': [], # vy front
                        # 'x21': [], # vy rear
                        'control0': [],
                        'control1': [],
                        'state': []}
        for i, agent in enumerate(self.agents):
            if DO_SCAN: observations['scans'].append(agent_scans[i])
            observations['poses_x'].append(agent.state[0])
            observations['poses_y'].append(agent.state[1])
            observations['poses_theta'].append(agent.state[4])
            observations['linear_vels_x'].append(agent.state[3])
            observations['linear_vels_y'].append(0.)
            observations['ang_vels_z'].append(agent.state[5])
            observations['x1'].append(agent.state[0])
            observations['x2'].append(agent.state[1])
            observations['x3'].append(agent.state[2])
            observations['x4'].append(agent.state[3])
            observations['x5'].append(agent.state[4])
            observations['x6'].append(agent.state[5])
            # observations['x11'].append(agent.state[10])
            # observations['x12'].append(agent.state[11])
            # observations['x13'].append(agent.state[12])
            # observations['x16'].append(agent.state[15])
            # observations['x21'].append(agent.state[20])
            observations['state'].append(agent.state)
            observations['control0'].append(control_inputs[i, 0])
            observations['control1'].append(control_inputs[i, 1])


        return observations

    def reset(self, initial_states):
        """
        Resets the simulation environment by given poses

        Arges:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            None
        """

        if initial_states.shape[0] != self.num_agents:
            raise ValueError('Number of poses for reset does not match number of agents.')
        
        if initial_states.shape[1] == 29:
            for i in range(self.num_agents):
                self.agents[i].reset(initial_states[i])

        else:
            # loop over poses to reset
            for i in range(self.num_agents):
                self.agents[i].reset(initial_states[i, [0, 1, 2]],
                                    steering_angle=initial_states[i, 3],
                                    velocity=initial_states[i, 4],
                                    yaw_rate=initial_states[i, 5],
                                    beta=initial_states[i, 6])
        self.observations = None
