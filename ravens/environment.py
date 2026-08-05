#!/usr/bin/env python

import os
import sys
import time
import threading
import pkg_resources

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from ravens.gripper import Gripper, Suction
from ravens import tasks, utils


class Environment():

    def __init__(
            self,
            disp=False,
            hz=240,
            deterministic=False,
            control_substeps=1,
            post_action_settle_steps=240):
        """Creates OpenAI gym-style env with support for PyBullet threading.

        Args:
            disp: Whether or not to use PyBullet's built-in display viewer.
                Use this either for local inspection of PyBullet, or when
                using any soft body (cloth or bags), because PyBullet's
                TinyRenderer graphics (used if disp=False) will make soft
                bodies invisible.
            hz: Parameter used in PyBullet to control the number of physics
                simulation steps. Higher values lead to more accurate physics
                at the cost of slower computaiton time. By default, PyBullet
                uses 240, but for soft bodies we need this to be at least 480
                to avoid cloth intersecting with the plane.
        """
        self.ee = None
        self.task = None
        self.plane_id = None
        self.workspace_id = None
        self.objects = []
        self.running = False
        self.fixed_objects = []
        self.pix_size = 0.003125
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.primitives = {'push':       self.push,
                           'sweep':      self.sweep,
                           'pick_place': self.pick_place,
                           'pick_probe_return': self.pick_probe_return,
                           'pick_planar_microprobe': self.pick_planar_microprobe,
                           'pick_precise_probe_return': self.pick_precise_probe_return,
                           'pick_precise_latch_probe': self.pick_precise_latch_probe,
                           'pick_precise_tension_extension': self.pick_precise_tension_extension}

        self._ccda_video_recorder = None
        self._ccda_video_label = ""
        self._ccda_motion_events = []
        # Phase3.12d-r1: serialize background physics with snapshot IO and
        # surface task-hook failures instead of silently killing the thread.
        self._ccda_step_lock = threading.RLock()
        self._ccda_physics_hook_error = None
        self._stop_event = threading.Event()

        # Set default movej timeout limit. For most tasks, 15 is reasonable.
        self.t_lim = 15

        # From Xuchen: need this for using any new deformable simulation.
        self.use_new_deformable = True
        self.hz = hz
        self.deterministic = bool(deterministic)
        self.control_substeps = int(control_substeps)
        self.post_action_settle_steps = int(post_action_settle_steps)
        if self.control_substeps <= 0:
            raise ValueError('control_substeps must be positive')
        if self.post_action_settle_steps < 0:
            raise ValueError('post_action_settle_steps must be non-negative')

        # Start PyBullet.
        p.connect(p.GUI if disp else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        assets_path = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(assets_path)
        # Fixed-step mode has no background thread, so the time step must be
        # installed before any explicit physics step is executed.
        p.setTimeStep(1.0 / self.hz)

        # Check PyBullet version (see also the cloth/bag task scripts!).
        p_version = pkg_resources.get_distribution('pybullet').version
        tested = ['2.8.4', '3.0.4']
        assert p_version in tested, f'PyBullet version {p_version} not in {tested}'

        # Move the camera a little closer to the scene. Most args are not used.
        # PyBullet defaults: yaw=50 and pitch=-35.
        if disp:
            _, _, _, _, _, _, _, _, _, _, _, target = p.getDebugVisualizerCamera()
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target,)

        # Control PyBullet simulation steps.
        self.step_thread = None
        if not self.deterministic:
            self.step_thread = threading.Thread(target=self.step_simulation)
            self.step_thread.daemon = True
            self.step_thread.start()

    def _step_physics_once_unlocked(self):
        """Execute exactly one PyBullet step and the CCDA task hooks."""
        task = getattr(self, 'task', None)
        pre_hook = getattr(task, 'physics_pre_step_hook', None)
        if callable(pre_hook):
            try:
                pre_hook()
            except Exception as exc:
                if self._ccda_physics_hook_error is None:
                    self._ccda_physics_hook_error = (
                        'physics_pre_step_hook: ' + repr(exc)
                    )

        p.stepSimulation()
        if self.ee is not None:
            self.ee.step()

        post_hook = getattr(task, 'physics_step_hook', None)
        if callable(post_hook):
            try:
                post_hook()
            except Exception as exc:
                if self._ccda_physics_hook_error is None:
                    self._ccda_physics_hook_error = (
                        'physics_step_hook: ' + repr(exc)
                    )

    def _raise_ccda_physics_hook_error(self):
        if self._ccda_physics_hook_error is not None:
            raise RuntimeError(self._ccda_physics_hook_error)

    def step_physics(self, steps=1):
        """Advance an exact number of physics steps.

        This is the only physics driver used by deterministic CCDA rollouts.
        In threaded mode it may be used only while the background loop is
        paused, preventing two independent callers from advancing PyBullet.
        """
        steps = int(steps)
        if steps < 0:
            raise ValueError('steps must be non-negative')
        if not self.deterministic and self.running:
            raise RuntimeError(
                'step_physics requires the threaded environment to be paused'
            )
        with self._ccda_step_lock:
            for _ in range(steps):
                self._step_physics_once_unlocked()
                self._raise_ccda_physics_hook_error()

    def wait_seconds(self, seconds):
        """Wait in threaded mode or advance an exact duration in fixed-step mode."""
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError('seconds must be non-negative')
        if self.deterministic:
            self.step_physics(int(round(seconds * self.hz)))
        else:
            time.sleep(seconds)

    def settle_for_seconds(self, seconds):
        """Settle a task for a deterministic or wall-clock duration."""
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError('seconds must be non-negative')
        if self.deterministic:
            self.step_physics(int(round(seconds * self.hz)))
        else:
            self.start()
            time.sleep(seconds)
            self.pause()

    def step_simulation(self):
        """Background physics loop used by the legacy threaded mode."""
        p.setTimeStep(1.0 / self.hz)
        while not self._stop_event.is_set():
            if self.running:
                with self._ccda_step_lock:
                    self._step_physics_once_unlocked()
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self._stop_event.set()
        if self.step_thread is not None and self.step_thread.is_alive():
            self.step_thread.join(timeout=1.0)
        with self._ccda_step_lock:
            if p.isConnected():
                p.disconnect()

    def start(self):
        self.running = True

    def pause(self):
        self.running = False

    def reset_ccda_runtime_after_restore(self):
        """Reset Python-side fields after ``pybullet.restoreState``."""
        self._ccda_physics_hook_error = None
        self._ccda_video_label = ""
        self._ccda_motion_events = []

        ee = self.ee
        if ee is None:
            return

        defaults = {
            "activated": False,
            "contact_constraint": None,
            "def_grip_item": None,
            "def_grip_anchors": None,
            "def_min_vertex": None,
            "def_min_vetex": None,
            "def_min_distance": None,
            "init_grip_distance": None,
            "init_grip_item": None,
        }
        for name, value in defaults.items():
            if hasattr(ee, name):
                setattr(ee, name, value)

    def set_ccda_video_recorder(self, recorder):
        """Attach or detach an optional CCDA simulation video recorder."""
        self._ccda_video_recorder = recorder
        self._ccda_video_label = ""

    def reset_ccda_motion_events(self):
        self._ccda_motion_events = []

    def ccda_motion_events(self):
        return [dict(event) for event in self._ccda_motion_events]

    def _record_ccda_hold_event(
            self,
            *,
            primitive,
            stage,
            physics_step_start,
            physics_step_end):
        self._ccda_motion_events.append({
            'primitive': str(primitive),
            'stage': str(stage),
            'label': str(stage),
            'physics_step_start': int(physics_step_start),
            'physics_step_end': int(physics_step_end),
            'physics_step_count': int(
                physics_step_end - physics_step_start
            ),
            'timeout_reason': None,
            'joint_timeout_count': 0,
            'joint_motion_success': True,
            'cartesian_endpoint_error': 0.0,
            'final_joint_error': [],
            'final_joint_error_max_abs': 0.0,
            'final_joint_error_norm': 0.0,
            'achieved_fraction': 1.0,
            'success': True,
            'event_kind': 'hold',
        })

    def record_ccda_frame(self, label=""):
        """Public frame-capture entry point for idle simulation phases."""
        self._ccda_record_frame(label)

    def _ccda_record_frame(self, label=""):
        recorder = getattr(self, '_ccda_video_recorder', None)
        if label and label != 'movej':
            self._ccda_video_label = label
        if recorder is None:
            return
        active_label = label
        if label == 'movej' and self._ccda_video_label:
            active_label = self._ccda_video_label
        with self._ccda_step_lock:
            recorder.record(active_label)

    def ccda_sensor_observation(self):
        """Robot-observable force proxies without hidden task labels."""
        joint_states = [
            p.getJointState(self.ur5, int(joint))
            for joint in self.joints
        ]
        joint_motor_torque = np.asarray(
            [float(state[3]) for state in joint_states],
            dtype=np.float64,
        )
        joint_reaction_force_torque = np.asarray(
            [np.asarray(state[2], dtype=np.float64) for state in joint_states],
            dtype=np.float64,
        )

        suction_force = np.zeros(3, dtype=np.float64)
        suction_torque = np.zeros(3, dtype=np.float64)
        constraint_available = False
        constraint_id = None

        ee = self.ee
        if ee is not None:
            value = getattr(ee, 'contact_constraint', None)
            if value is not None:
                constraint_id = int(value)
                try:
                    state = np.asarray(
                        p.getConstraintState(constraint_id),
                        dtype=np.float64,
                    ).reshape(-1)
                    if state.size >= 3:
                        suction_force[:] = state[:3]
                        constraint_available = True
                    if state.size >= 6:
                        suction_torque[:] = state[3:6]
                except Exception:
                    constraint_available = False

        grasp_active = bool(
            ee is not None and getattr(ee, 'activated', False)
        )
        return {
            'joint_motor_torque': [float(v) for v in joint_motor_torque],
            'joint_motor_torque_norm': float(np.linalg.norm(joint_motor_torque)),
            'joint_reaction_force_torque': (
                joint_reaction_force_torque.astype(float).tolist()
            ),
            'joint_reaction_force_torque_norm': float(
                np.linalg.norm(joint_reaction_force_torque)
            ),
            'suction_force_xyz': [float(v) for v in suction_force],
            'suction_force_norm': float(np.linalg.norm(suction_force)),
            'suction_torque_xyz': [float(v) for v in suction_torque],
            'suction_torque_norm': float(np.linalg.norm(suction_torque)),
            'grasp_active': int(grasp_active),
            'constraint_available': int(constraint_available),
            'constraint_id': constraint_id,
        }

    def is_static(self):
        """Checks if env is static, used for checking if action finished.

        However, this won't work in PyBullet (at least v2.8.4) since soft
        bodies cause this code to hang. Therefore, look at the task's
        `def_IDs` list, which by design will have all IDs of soft bodies.
        Furthermore, for the bag tasks, the beads generally move around, so
        for those, just use a hard cutoff limit (outside this method).
        """
        if self.is_softbody_env():
            assert len(self.task.def_IDs) > 0, 'Did we forget to add to def_IDs?'
            v = [np.linalg.norm(p.getBaseVelocity(i)[0]) for i in self.objects
                    if i not in self.task.def_IDs]
        else:
            v = [np.linalg.norm(p.getBaseVelocity(i)[0]) for i in self.objects]
        return all(np.array(v) < 1e-2)

    def add_object(self, urdf, pose, fixed=False):
        fixedBase = 1 if fixed else 0
        object_id = p.loadURDF(urdf, pose[0], pose[1], useFixedBase=fixedBase)
        if fixed:
            self.fixed_objects.append(object_id)
        else:
            self.objects.append(object_id)
        return object_id

    #-------------------------------------------------------------------------
    # Standard Gym Functions
    #-------------------------------------------------------------------------

    def reset(self, task, last_info=None, disable_render_load=True):
        """Sets up PyBullet, loads models, resets the specific task.

        We do a step() call with act=None at the end. This will only return
        an empty obs dict, obs={}. For some tasks where the reward could be
        nonzero at the start, we can report the reward shown here.

        Args:
            last_info: Only for goal-conditioned learning DURING TEST TIME,
                since we load in a target image, but we also want to load in
                final object poses, since in many cases we get better
                accuracy. For simplicity, I suggest we put all this in the
                `info` dict, and we load it as the `info` that happens after
                finishing. That will have the most up to date object poses,
                and we can always add extra information as needed in
                last_info['extras'].
            disable_render_load: Need this as True to avoid `p.loadURDF`
                becoming a time bottleneck, judging from my profiling.
        """
        self.pause()
        self._ccda_physics_hook_error = None
        self.task = task
        self.objects = []
        self.fixed_objects = []
        if self.use_new_deformable:
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        else:
            p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Slightly increase default movej timeout for the more demanding tasks.
        if self.is_bag_env():
            self.t_lim = 60
            if isinstance(self.task, tasks.names['bag-color-goal']):
                self.t_lim = 120

        # Empirically, this seems to make loading URDFs faster w/remote displays.
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        id_plane = p.loadURDF('assets/plane/plane.urdf', [0, 0, -0.001])
        id_ws = p.loadURDF('assets/ur5/workspace.urdf', [0.5, 0, 0])
        self.plane_id = int(id_plane)
        self.workspace_id = int(id_ws)

        # Load UR5 robot arm equipped with task-specific end effector.
        self.ur5 = p.loadURDF(f'assets/ur5/ur5-{self.task.ee}.urdf')
        self.ee_tip_link = 12
        if self.task.ee == 'suction':
            self.ee = Suction(self.ur5, 11)
        elif self.task.ee == 'gripper':
            self.ee = Robotiq2F85(self.ur5, 9)
            self.ee_tip_link = 10
        else:
            self.ee = Gripper()

        # Get revolute joint indices of robot (skip fixed joints).
        num_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(num_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        # Get end effector tip pose in home configuration.
        ee_tip_state = p.getLinkState(self.ur5, self.ee_tip_link)
        self.home_pose = np.array(ee_tip_state[0] + ee_tip_state[1])

        # Reset end effector.
        self.ee.release()

        # Seems like this should be BEFORE reset()
        # since for bag-items we may assign to True!
        task.exit_gracefully = False

        # Reset task.
        if last_info is not None:
            task.reset(self, last_info)
        else:
            task.reset(self)

        # Daniel: might be useful to have this debugging tracker.
        self.IDTracker = utils.TrackIDs()
        self.IDTracker.add(id_plane, 'Plane')
        self.IDTracker.add(id_ws, 'Workspace')
        self.IDTracker.add(self.ur5, 'UR5')
        try:
            self.IDTracker.add(self.ee.body, 'Gripper.body')
        except:
            pass

        # Daniel: add other IDs, but not all envs use the ID tracker.
        try:
            task_IDs = task.get_ID_tracker()
            for i in task_IDs:
                self.IDTracker.add(i, task_IDs[i])
        except AttributeError:
            pass
        #print(self.IDTracker)  # If doing multiple episodes, check if I reset the ID dict!
        assert id_ws == 1, f'Workspace ID: {id_ws}'

        # Daniel: tune gripper for deformables if applicable, and CHECK HZ!!
        if self.is_softbody_env():
            self.ee.set_def_threshold(threshold=self.task.def_threshold)
            self.ee.set_def_nb_anchors(nb_anchors=self.task.def_nb_anchors)
            assert self.hz >= 480, f'Error, hz={self.hz} is too small!'

        # Restart simulation.
        self.start()
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        (obs, _, _, _) = self.step()
        defer_hidden_factor = (
            os.environ.get(
                'CCDA_DEFER_HIDDEN_FACTOR_ARMING',
                os.environ.get('CCDA_DEFER_HIDDEN_FRICTION_ARMING', '0'),
            ) == '1'
        )
        arm_hidden_factor = getattr(
            self.task,
            'arm_ccda_hidden_factor_after_settle',
            None,
        )
        if callable(arm_hidden_factor) and not defer_hidden_factor:
            self.pause()
            arm_hidden_factor()
            self.start()
        return obs

    def step(self, act=None):
        """Execute action with specified primitive.

        For each episode (training, loading, etc.), this is normally called the first
        time from `env.reset()` above, with NO action, and returns an EMPTY observation.
        Then, it's called a SECOND time with an action that lacks a primitive (set to
        None, even though the key exists). But, this method will return an actual image
        observation that we pass to the policy. Finally, subsequent calls will have
        proper actions.

        (Sept 08) Added graceful exit functionality. This will let the code terminate
        early but with task.done=False as we didn't actually 'finish' -- just failed.
        Should use task.done=True to indicate any form of 'successful' dones (hitting
        time limit doesn't count).

        (Oct 09) Clarify documentation for the confusing first time step. I now see
        with ground truth agents, there is no 'second action lacking a primitive',
        because ground truth agents don't need images (see their `act` method).
        """
        action_executed = bool(act and act['primitive'])
        if action_executed:
            success = self.primitives[act['primitive']](**act['params'])

            # Exit early if action failed. Daniel: adding exit_gracefully.
            if (not success) or self.task.exit_gracefully:
                _, reward_extras = self.task.reward()
                info = self.info
                reward_extras['task.done'] = False

                # Means we hit irrecoverable action, exit now (reset to False!!).
                if self.task.exit_gracefully:
                    reward_extras['exit_gracefully'] = True
                    self.task.exit_gracefully = False  # important !!!

                # For consistency?
                if isinstance(self.task, tasks.names['cloth-flat-notarget']):
                    info['sampled_zone_pose'] = self.task.zone_pose
                elif isinstance(self.task, tasks.names['bag-color-goal']):
                    info['bag_base_pos'] = self.task.bag_base_pos[0]
                    info['bag_base_orn'] = self.task.bag_base_orn[0]
                    info['bag_target_color'] = self.task.bag_colors[0]

                info['extras'] = reward_extras
                return {}, 0, True, info

        if self.deterministic:
            # Do not wait on a velocity predicate: hidden contact conditions
            # can reach that predicate at different times. A fixed number of
            # post-action steps gives repeatable phase boundaries.
            if action_executed and self.post_action_settle_steps:
                self.step_physics(self.post_action_settle_steps)
        else:
            start_t = time.time()
            while not self.is_static():
                self._ccda_record_frame("settle")
                if self.is_bag_env() and (time.time() - start_t > 2.0):
                    break
                time.sleep(0.001)

        # Compute task rewards.
        reward, reward_extras = self.task.reward()
        done = self.task.done()

        # Pass ground truth robot state as info.
        info = self.info

        # Daniel: fine-grained info about rewards (since it's nuanced for some tasks).
        # If we hit time limit, `task.done` will check if we succeeded on last action.
        reward_extras['task.done'] = done
        info['extras'] = reward_extras
        if isinstance(self.task, tasks.names['cloth-flat-notarget']):
            info['sampled_zone_pose'] = self.task.zone_pose
        elif isinstance(self.task, tasks.names['bag-color-goal']):
            info['bag_base_pos'] = self.task.bag_base_pos[0]
            info['bag_base_orn'] = self.task.bag_base_orn[0]
            info['bag_target_color'] = self.task.bag_colors[0]

        # Get camera observations per specified config.
        obs = {}
        if act and 'camera_config' in act:
            obs['color'], obs['depth'] = [], []
            for config in act['camera_config']:
                color, depth, _ = self.render(config)
                obs['color'].append(color)
                obs['depth'].append(depth)

        return obs, reward, done, info

    def render(self, config):
        """Render RGB-D image with specified configuration."""

        # Compute OpenGL camera settings.
        lookdir = np.array([0, 0, 1]).reshape(3, 1)
        updir = np.array([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.array(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_length = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (np.arctan((config['image_size'][0] /
                           2) / focal_length) * 2 / np.pi) * 180

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config['image_size'][1],
            height=config['image_size'][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (config['image_size'][0],
                            config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        color_image_size = (color_image_size[0], color_image_size[1], 3)
        if config['noise']:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color_image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += np.random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    @property
    def info(self):
        """Normally returns dict of:

            object id : (position, rotation, dimensions)

        However, I made a few changes. First, removing IDs in some tasks, so
        we shouldn't query their position. Second, adding object mesh for
        gt_state and soft bodies. The second `if` test is redundant but just
        in case. Here, we instead map to the mesh data instead of (position,
        rotation, dimension). This is: (nb_vertices, (positions)) where the
        latter is a tuple that has all the 3D positions of each vertex in the
        simulation mesh, e.g., it's 100 for cloth.

        Note on soft body IDs:
        cloth-cover: ID 5 is cloth (ID 4 is item to cover)
        cloth-flat(notarget): ID 5 is cloth (ID 4 is the zone, though in the no
            target case we remove it ... historical reasons).
        bag tasks: all have ID 5 as the bag, because they use ID 4 for the zone,
            (with bag-alone removing zone) and other items are added afterwards.

        To see how we use the special case for soft bodies, see:
            ravens/agents/gt_state.py and the extraact_x_y_theta method.
        We depend on assuming that len(info[id]) = 2 instead of 3.
        """
        removed_IDs = []
        if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
                isinstance(self.task, tasks.names['bag-alone-open'])):
            removed_IDs.append(self.task.zone_ID)

        # Daniel: special case for soft bodies (and gt_state). For now only cloth.
        softbody_id = -1
        if self.is_cloth_env():
            assert len(self.task.def_IDs) == 1, self.task.def_IDs
            softbody_id = self.task.def_IDs[0]

        # object id : (position, rotation, dimensions)
        info = {}
        for object_id in (self.fixed_objects + self.objects):
            if object_id in removed_IDs:
                continue

            # Daniel: special cases for soft bodies (and gt_state), and then bags.
            if (object_id == softbody_id) and self.is_cloth_env():
                info[object_id] = p.getMeshData(object_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
            elif isinstance(self.task, tasks.names['bag-color-goal']):
                # Note: we don't do this for sorting? :X
                position, rotation = p.getBasePositionAndOrientation(object_id)
                dimensions = p.getVisualShapeData(object_id)[0][3]
                rgba_color = p.getVisualShapeData(object_id)[0][7]  # see pybullet docs
                assert rgba_color[3] == 1, rgba_color
                rgb_color = rgba_color[0:3]  # we can ignore the last component it is always 1
                info[object_id] = (position, rotation, dimensions, rgb_color)
            else:
                # The usual case.
                position, rotation = p.getBasePositionAndOrientation(object_id)
                dimensions = p.getVisualShapeData(object_id)[0][3]
                info[object_id] = (position, rotation, dimensions)

        return info

    #-------------------------------------------------------------------------
    # Robot Movement Functions
    #-------------------------------------------------------------------------

    def _movej_fixed(self, targj, speed=0.01, t_lim=20, joint_tolerance=None):
        targj = np.asarray(targj, dtype=np.float64)
        speed = float(speed)
        tolerance = 1e-2 if joint_tolerance is None else float(joint_tolerance)
        if speed <= 0:
            raise ValueError('movej speed must be positive')
        if tolerance <= 0:
            raise ValueError('joint_tolerance must be positive')

        max_physics_steps = max(1, int(round(float(t_lim) * self.hz)))
        max_iterations = max(
            1,
            int(np.ceil(max_physics_steps / self.control_substeps)),
        )
        for _ in range(max_iterations):
            currj = np.asarray(
                [p.getJointState(self.ur5, i)[0] for i in self.joints],
                dtype=np.float64,
            )
            diffj = targj - currj
            if np.all(np.abs(diffj) < tolerance):
                return True
            norm = float(np.linalg.norm(diffj))
            velocity = diffj / norm if norm > 0 else np.zeros_like(diffj)
            stepj = currj + velocity * min(speed, norm)
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=np.ones(len(self.joints)),
            )
            self._ccda_record_frame('movej')
            self.step_physics(self.control_substeps)

        print('Warning: deterministic movej exceeded {} physics steps.'.format(max_physics_steps))
        return False

    def movej(self, targj, speed=0.01, t_lim=20, joint_tolerance=None):
        tolerance = 1e-2 if joint_tolerance is None else float(joint_tolerance)
        if tolerance <= 0:
            raise ValueError('joint_tolerance must be positive')
        if self.deterministic:
            return self._movej_fixed(
                targj,
                speed=speed,
                t_lim=t_lim,
                joint_tolerance=tolerance,
            )

        t0 = time.time()
        while time.time() - t0 < t_lim:
            currj = np.asarray(
                [p.getJointState(self.ur5, i)[0] for i in self.joints],
                dtype=np.float64,
            )
            diffj = np.asarray(targj) - currj
            if np.all(np.abs(diffj) < tolerance):
                return True
            norm = float(np.linalg.norm(diffj))
            velocity = diffj / norm if norm > 0 else 0
            stepj = currj + velocity * min(speed, norm)
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=np.ones(len(self.joints)),
            )
            self._ccda_record_frame('movej')
            time.sleep(0.001)
        print('Warning: movej exceeded {} sec timeout.'.format(t_lim))
        return False

    def movep(self, pose, speed=0.01, joint_tolerance=None):
        targj = self.solve_IK(pose)
        return self.movej(
            targj,
            speed,
            self.t_lim,
            joint_tolerance=joint_tolerance,
        )

    def movep_precise(
            self,
            pose,
            speed=0.001,
            joint_tolerance=1e-4,
            cartesian_tolerance=2e-4,
            max_corrections=3,
            label='precise_move',
            primitive='pick_precise_probe_return',
            record_event=True):
        if not self.deterministic:
            raise RuntimeError('movep_precise requires deterministic execution')

        target = np.asarray(pose, dtype=np.float64).reshape(-1)
        if target.size != 7:
            raise ValueError('pose must contain XYZ plus quaternion')
        if joint_tolerance <= 0 or cartesian_tolerance <= 0:
            raise ValueError('tolerances must be positive')
        max_corrections = int(max_corrections)
        if max_corrections <= 0:
            raise ValueError('max_corrections must be positive')
        record_event = bool(record_event)

        before = np.asarray(
            p.getLinkState(
                self.ur5,
                self.ee_tip_link,
                computeForwardKinematics=True,
            )[0],
            dtype=np.float64,
        )
        task = getattr(self, 'task', None)
        physics_counter = getattr(task, 'physics_step_count', None)
        physics_step_start = (
            int(physics_counter()) if callable(physics_counter) else None
        )
        joint_motion_success = True
        joint_timeout_count = 0
        after = before.copy()
        endpoint_error = float(np.linalg.norm(target[:3] - before))
        corrections_used = 0
        command_target = target.copy()

        for correction in range(max_corrections):
            corrections_used = correction + 1
            correction_success = self.movep(
                command_target,
                speed=speed,
                joint_tolerance=joint_tolerance,
            )
            joint_motion_success &= correction_success
            if not correction_success:
                joint_timeout_count += 1
            after = np.asarray(
                p.getLinkState(
                    self.ur5,
                    self.ee_tip_link,
                    computeForwardKinematics=True,
                )[0],
                dtype=np.float64,
            )
            endpoint_error = float(np.linalg.norm(target[:3] - after))
            if endpoint_error <= cartesian_tolerance:
                break
            command_target[:3] += target[:3] - after

        joints = list(getattr(self, 'joints', []))
        if joints:
            final_joint_target = np.asarray(
                self.solve_IK(command_target), dtype=np.float64
            )
            final_joint_position = np.asarray(
                [p.getJointState(self.ur5, i)[0] for i in joints],
                dtype=np.float64,
            )
            final_joint_error = final_joint_target - final_joint_position
            final_joint_error_max_abs = float(
                np.max(np.abs(final_joint_error))
            )
            final_joint_error_norm = float(np.linalg.norm(final_joint_error))
        else:
            final_joint_error = np.asarray([], dtype=np.float64)
            final_joint_error_max_abs = None
            final_joint_error_norm = None
        endpoint_reached = bool(endpoint_error <= cartesian_tolerance)
        if joint_timeout_count and endpoint_reached:
            timeout_reason = 'joint_timeout_recovered_by_cartesian_endpoint'
        elif joint_timeout_count:
            timeout_reason = 'joint_timeout_and_cartesian_endpoint_miss'
        elif not endpoint_reached:
            timeout_reason = 'cartesian_endpoint_miss'
        else:
            timeout_reason = None
        physics_step_end = (
            int(physics_counter()) if callable(physics_counter) else None
        )

        requested_vector = target[:3] - before
        achieved_vector = after - before
        requested_distance = float(np.linalg.norm(requested_vector))
        achieved_distance = float(np.linalg.norm(achieved_vector))
        if requested_distance > 1e-12:
            achieved_projection = float(
                np.dot(achieved_vector, requested_vector / requested_distance)
            )
            achieved_fraction = achieved_projection / requested_distance
        else:
            achieved_projection = 0.0
            achieved_fraction = 1.0

        event = {
            'primitive': str(primitive),
            'stage': str(label),
            'label': str(label),
            'requested_position': target[:3].astype(float).tolist(),
            'before_position': before.astype(float).tolist(),
            'after_position': after.astype(float).tolist(),
            'requested_distance': requested_distance,
            'achieved_distance': achieved_distance,
            'achieved_projection': achieved_projection,
            'achieved_fraction': float(achieved_fraction),
            'endpoint_error': endpoint_error,
            'cartesian_endpoint_error': endpoint_error,
            'final_joint_error': final_joint_error.astype(float).tolist(),
            'final_joint_error_max_abs': final_joint_error_max_abs,
            'final_joint_error_norm': final_joint_error_norm,
            'joint_tolerance': float(joint_tolerance),
            'cartesian_tolerance': float(cartesian_tolerance),
            'corrections_used': int(corrections_used),
            'joint_motion_success': bool(joint_motion_success),
            'joint_timeout_count': int(joint_timeout_count),
            'timeout_reason': timeout_reason,
            'physics_step_start': physics_step_start,
            'physics_step_end': physics_step_end,
            'physics_step_count': (
                physics_step_end - physics_step_start
                if physics_step_start is not None and physics_step_end is not None
                else None
            ),
            # The primitive is Cartesian. A fixed-step joint timeout is retained
            # above as audit evidence, but is locally recoverable when residual
            # feedback has placed the endpoint inside the unchanged tolerance.
            'success': endpoint_reached,
        }
        if record_event:
            self._ccda_motion_events.append(event)
        return bool(event['success'])

    def solve_IK(self, pose):
        homej_list = np.array(self.homej).tolist()
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip_link,
            targetPosition=pose[:3],
            targetOrientation=pose[3:],
            lowerLimits=[-17, -2.3562, -17, -17, -17, -17],
            upperLimits=[17, 0, 17, 17, 17, 17],
            jointRanges=[17] * 6,
            restPoses=homej_list,
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.array(joints)
        joints[joints > 2 * np.pi] = joints[joints > 2 * np.pi] - 2 * np.pi
        joints[joints < -2 * np.pi] = joints[joints < -2 * np.pi] + 2 * np.pi
        return joints

    #-------------------------------------------------------------------------
    # Motion Primitives
    #-------------------------------------------------------------------------

    def pick_place(self, pose0, pose1):
        """Execute pick and place primitive.

        Standard ravens tasks use the `delta` vector to lower the gripper
        until it makes contact with something. With deformables, however, we
        need to consider cases when the gripper could detect a rigid OR a
        soft body (cloth or bag); it should grip the first item it touches.
        This is handled in the Gripper class.

        Different deformable ravens tasks use slightly different parameters
        for better physics (and in some cases, faster simulation). Therefore,
        rather than make special cases here, those tasks will define their
        own action parameters, which we use here if they exist. Otherwise, we
        stick to defaults from standard ravens. Possible action parameters a
        task might adjust:

            speed: how fast the gripper moves.
            delta_z: how fast the gripper lowers for picking / placing.
            prepick_z: height of the gripper when it goes above the target
                pose for picking, just before it lowers.
            postpick_z: after suction gripping, raise to this height, should
                generally be low for cables / cloth.
            preplace_z: like prepick_z, but for the placing pose.
            pause_place: add a small pause for some tasks (e.g., bags) for
                slightly better soft body physics.
            final_z: height of the gripper after the action. Recommended to
                leave it at the default of 0.3, because it has to be set high
                enough to avoid the gripper occluding the workspace when
                generating color/depth maps.
        Args:
            pose0: picking pose.
            pose1: placing pose.

        Returns:
            A bool indicating whether the action succeeded or not, via
            checking the sequence of movep calls. If any movep failed, then
            self.step() will terminate the episode after this action.
        """
        # Defaults used in the standard Ravens environments.
        speed = 0.01
        delta_z = -0.001
        prepick_z = 0.3
        postpick_z = 0.3
        preplace_z = 0.3
        pause_place = 0.0
        final_z = 0.3

        # Find parameters, which may depend on the task stage.
        if hasattr(self.task, 'primitive_params'):
            ts = self.task.task_stage
            if 'prepick_z' in self.task.primitive_params[ts]:
                prepick_z = self.task.primitive_params[ts]['prepick_z']
            speed       = self.task.primitive_params[ts]['speed']
            delta_z     = self.task.primitive_params[ts]['delta_z']
            postpick_z  = self.task.primitive_params[ts]['postpick_z']
            preplace_z  = self.task.primitive_params[ts]['preplace_z']
            pause_place = self.task.primitive_params[ts]['pause_place']

        # Used to track deformable IDs, so that we can get the vertices.
        def_IDs = []
        if hasattr(self.task, 'def_IDs'):
            def_IDs = self.task.def_IDs

        # Otherwise, proceed as normal.
        success = True
        pick_position = np.array(pose0[0])
        pick_rotation = np.array(pose0[1])
        prepick_position = pick_position.copy()
        prepick_position[2] = prepick_z

        # Execute picking motion primitive.
        prepick_pose = np.hstack((prepick_position, pick_rotation))
        self._ccda_record_frame("prepick")
        success &= self.movep(prepick_pose)
        target_pose = prepick_pose.copy()
        delta = np.array([0, 0, delta_z, 0, 0, 0, 0])

        # Lower gripper until (a) touch object (rigid OR softbody), or (b) hit ground.
        self._ccda_record_frame("lower")
        while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
            target_pose += delta
            success &= self.movep(target_pose)

        # Create constraint (rigid objects) or anchor (deformable).
        self._ccda_record_frame("grasp")
        self.ee.activate(self.objects, def_IDs)
        self._ccda_record_frame("grasp")

        # Increase z slightly (or hard-code it) and check picking success.
        self._ccda_record_frame("lift")
        if self.is_softbody_env() or self.is_new_cable_env():
            prepick_pose[2] = postpick_z
            success &= self.movep(prepick_pose, speed=speed)
            self.wait_seconds(pause_place) # extra rest for bags
        elif isinstance(self.task, tasks.names['cable']):
            prepick_pose[2] = 0.03
            success &= self.movep(prepick_pose, speed=0.001)
        else:
            prepick_pose[2] += pick_position[2]
            success &= self.movep(prepick_pose)
        pick_success = self.ee.check_grasp()

        if pick_success:
            place_position = np.array(pose1[0])
            place_rotation = np.array(pose1[1])
            preplace_position = place_position.copy()
            preplace_position[2] = 0.3 + pick_position[2]

            # Execute placing motion primitive if pick success.
            preplace_pose = np.hstack((preplace_position, place_rotation))
            self._ccda_record_frame("preplace")
            if self.is_softbody_env() or self.is_new_cable_env():
                preplace_pose[2] = preplace_z
                success &= self.movep(preplace_pose, speed=speed)
                self.wait_seconds(pause_place) # extra rest for bags
            elif isinstance(self.task, tasks.names['cable']):
                preplace_pose[2] = 0.03
                success &= self.movep(preplace_pose, speed=0.001)
            else:
                success &= self.movep(preplace_pose)

            # Lower the gripper. Here, we have a fixed speed=0.01. TODO: consider additional
            # testing with bags, so that the 'lowering' process for bags is more reliable.
            target_pose = preplace_pose.copy()
            self._ccda_record_frame("lower")
            while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
                target_pose += delta
                success &= self.movep(target_pose)

            # Release AND get gripper high up, to clear the view for images.
            self._ccda_record_frame("release")
            self.ee.release()
            preplace_pose[2] = final_z
            success &= self.movep(preplace_pose)
            self._ccda_record_frame("settle")
        else:
            # Release AND get gripper high up, to clear the view for images.
            self._ccda_record_frame("release")
            self.ee.release()
            prepick_pose[2] = final_z
            success &= self.movep(prepick_pose)
            self._ccda_record_frame("settle")
        return success

    def pick_probe_return(
            self,
            pose0,
            pose_probe,
            pose_return,
            hold_steps=0):
        """Pick a cable endpoint, probe outward, return, then release.

        This primitive is intended only for deterministic CCDA calibration. The
        same grasp remains active during outward and return motion, avoiding a
        condition-dependent re-grasp between the two waypoints.
        """
        if not self.deterministic:
            raise RuntimeError(
                'pick_probe_return requires deterministic fixed-step execution'
            )

        hold_steps = int(hold_steps)
        if hold_steps < 0:
            raise ValueError('hold_steps must be non-negative')

        speed = 0.01
        delta_z = -0.001
        prepick_z = 0.3
        postpick_z = 0.3
        preplace_z = 0.3
        final_z = 0.3

        if hasattr(self.task, 'primitive_params'):
            stage = self.task.task_stage
            params = self.task.primitive_params[stage]
            prepick_z = params.get('prepick_z', prepick_z)
            speed = params['speed']
            delta_z = params['delta_z']
            postpick_z = params['postpick_z']
            preplace_z = params['preplace_z']

        deformable_ids = []
        if hasattr(self.task, 'def_IDs'):
            deformable_ids = self.task.def_IDs

        success = True
        pick_position = np.asarray(pose0[0], dtype=np.float64)
        pick_rotation = np.asarray(pose0[1], dtype=np.float64)
        prepick_position = pick_position.copy()
        prepick_position[2] = prepick_z
        prepick_pose = np.hstack((prepick_position, pick_rotation))

        self._ccda_record_frame('probe_prepick')
        success &= self.movep(prepick_pose)

        target_pose = prepick_pose.copy()
        delta = np.asarray([0, 0, delta_z, 0, 0, 0, 0], dtype=np.float64)
        self._ccda_record_frame('probe_lower')
        while (
                not self.ee.detect_contact(deformable_ids)
                and target_pose[2] > 0):
            target_pose += delta
            success &= self.movep(target_pose)

        self._ccda_record_frame('probe_grasp')
        self.ee.activate(self.objects, deformable_ids)

        prepick_pose[2] = postpick_z
        self._ccda_record_frame('probe_lift')
        success &= self.movep(prepick_pose, speed=speed)
        if not self.ee.check_grasp():
            self.ee.release()
            prepick_pose[2] = final_z
            success &= self.movep(prepick_pose)
            return False

        def waypoint_pose(value):
            position = np.asarray(value[0], dtype=np.float64).copy()
            rotation = np.asarray(value[1], dtype=np.float64)
            position[2] = preplace_z
            return np.hstack((position, rotation))

        probe_pose = waypoint_pose(pose_probe)
        return_pose = waypoint_pose(pose_return)

        self._ccda_record_frame('probe_out')
        success &= self.movep(probe_pose, speed=speed)

        if hold_steps:
            self._ccda_record_frame('probe_hold')
            self.step_physics(hold_steps)

        self._ccda_record_frame('probe_return')
        success &= self.movep(return_pose, speed=speed)

        target_pose = return_pose.copy()
        self._ccda_record_frame('probe_return_lower')
        while (
                not self.ee.detect_contact(deformable_ids)
                and target_pose[2] > 0):
            target_pose += delta
            success &= self.movep(target_pose)

        self._ccda_record_frame('probe_release')
        self.ee.release()
        return_pose[2] = final_z
        success &= self.movep(return_pose)
        self._ccda_record_frame('probe_settle')
        return bool(success)

    def pick_planar_microprobe(
            self,
            pose0,
            pose_probe,
            pose_return,
            lift_height=0.0015,
            hold_steps=60,
            return_hold_steps=120,
            post_release_steps=120,
            approach_height=0.02,
            retreat_z=0.3):
        """Low-height planar probe with one continuous grasp."""
        if not self.deterministic:
            raise RuntimeError(
                'pick_planar_microprobe requires deterministic execution'
            )

        lift_height = float(lift_height)
        approach_height = float(approach_height)
        retreat_z = float(retreat_z)
        hold_steps = int(hold_steps)
        return_hold_steps = int(return_hold_steps)
        post_release_steps = int(post_release_steps)

        if lift_height <= 0:
            raise ValueError('lift_height must be positive')
        if approach_height <= lift_height:
            raise ValueError('approach_height must exceed lift_height')
        if retreat_z <= 0:
            raise ValueError('retreat_z must be positive')
        if min(hold_steps, return_hold_steps, post_release_steps) < 0:
            raise ValueError('microprobe step counts must be non-negative')

        speed = 0.001
        delta_z = -0.0005
        if hasattr(self.task, 'primitive_params'):
            params = self.task.primitive_params[self.task.task_stage]
            speed = float(params.get('speed', speed))
            delta_z = float(params.get('delta_z', delta_z))
        if delta_z >= 0:
            raise ValueError('microprobe delta_z must be negative')

        deformable_ids = getattr(self.task, 'def_IDs', [])
        pick_position = np.asarray(pose0[0], dtype=np.float64)
        pick_rotation = np.asarray(pose0[1], dtype=np.float64)
        probe_position = np.asarray(pose_probe[0], dtype=np.float64)
        return_position = np.asarray(pose_return[0], dtype=np.float64)

        success = True
        approach_pose = np.hstack((
            [pick_position[0], pick_position[1], pick_position[2] + approach_height],
            pick_rotation,
        ))
        self._ccda_record_frame('micro_approach')
        success &= self.movep(approach_pose, speed=speed)

        target_pose = approach_pose.copy()
        floor_limit = max(0.0, float(pick_position[2]) - 0.01)
        self._ccda_record_frame('micro_lower')
        while not self.ee.detect_contact(deformable_ids) and target_pose[2] > floor_limit:
            target_pose[2] += delta_z
            success &= self.movep(target_pose, speed=speed)
            if not success:
                return False

        self._ccda_record_frame('micro_grasp')
        self.ee.activate(self.objects, deformable_ids)
        if not self.ee.check_grasp():
            self.ee.release()
            retreat_pose = approach_pose.copy()
            retreat_pose[2] = retreat_z
            self.movep(retreat_pose, speed=speed)
            return False

        current_tip = p.getLinkState(
            self.ur5,
            self.ee_tip_link,
            computeForwardKinematics=True,
        )[0]
        probe_z = float(current_tip[2]) + lift_height

        lift_pose = np.hstack((
            [pick_position[0], pick_position[1], probe_z],
            pick_rotation,
        ))
        self._ccda_record_frame('micro_lift')
        success &= self.movep(lift_pose, speed=speed)

        outward_pose = np.hstack((
            [probe_position[0], probe_position[1], probe_z],
            pick_rotation,
        ))
        self._ccda_record_frame('micro_probe_out')
        success &= self.movep(outward_pose, speed=speed)
        if hold_steps:
            self._ccda_record_frame('micro_probe_hold')
            self.step_physics(hold_steps)

        return_pose = np.hstack((
            [return_position[0], return_position[1], probe_z],
            pick_rotation,
        ))
        self._ccda_record_frame('micro_probe_return')
        success &= self.movep(return_pose, speed=speed)
        if return_hold_steps:
            self._ccda_record_frame('micro_return_hold')
            self.step_physics(return_hold_steps)

        release_pose = return_pose.copy()
        release_pose[2] = float(pick_position[2])
        self._ccda_record_frame('micro_return_lower')
        success &= self.movep(release_pose, speed=speed)
        self._ccda_record_frame('micro_release')
        self.ee.release()
        if post_release_steps:
            self._ccda_record_frame('micro_post_release')
            self.step_physics(post_release_steps)

        retreat_pose = release_pose.copy()
        retreat_pose[2] = retreat_z
        self._ccda_record_frame('micro_retreat')
        success &= self.movep(retreat_pose, speed=speed)
        self._ccda_record_frame('micro_done')
        return bool(success)

    def pick_precise_probe_return(
            self,
            pose0,
            pose_probe,
            pose_return,
            lift_height=0.002,
            hold_steps=60,
            return_hold_steps=120,
            post_release_steps=120,
            approach_height=0.02,
            retreat_z=0.3,
            joint_tolerance=1e-4,
            cartesian_tolerance=2e-4,
            min_achieved_fraction=0.8):
        if not self.deterministic:
            raise RuntimeError('pick_precise_probe_return requires fixed-step mode')
        if lift_height <= 0:
            raise ValueError('lift_height must be positive')
        if min(hold_steps, return_hold_steps, post_release_steps) < 0:
            raise ValueError('hold steps must be non-negative')

        speed = 0.001
        delta_z = -0.0005
        if hasattr(self.task, 'primitive_params'):
            params = self.task.primitive_params[self.task.task_stage]
            speed = float(params.get('speed', speed))
            delta_z = float(params.get('delta_z', delta_z))
        if delta_z >= 0:
            raise ValueError('delta_z must be negative')

        deformable_ids = getattr(self.task, 'def_IDs', [])
        pick = np.asarray(pose0[0], dtype=np.float64)
        rotation = np.asarray(pose0[1], dtype=np.float64)
        probe = np.asarray(pose_probe[0], dtype=np.float64)
        returned = np.asarray(pose_return[0], dtype=np.float64)

        success = True
        approach = np.hstack(([pick[0], pick[1], pick[2] + approach_height], rotation))
        success &= self.movep(approach, speed=speed)

        lower = approach.copy()
        floor_limit = max(0.0, float(pick[2]) - 0.01)
        while not self.ee.detect_contact(deformable_ids) and lower[2] > floor_limit:
            lower[2] += delta_z
            success &= self.movep(lower, speed=speed, joint_tolerance=joint_tolerance)
            if not success:
                return False

        self.ee.activate(self.objects, deformable_ids)
        if not self.ee.check_grasp():
            self.ee.release()
            return False

        tip_position = np.asarray(
            p.getLinkState(
                self.ur5,
                self.ee_tip_link,
                computeForwardKinematics=True,
            )[0],
            dtype=np.float64,
        )
        probe_z = float(tip_position[2]) + float(lift_height)

        lift = np.hstack((
            [tip_position[0], tip_position[1], probe_z],
            rotation,
        ))
        success &= self.movep_precise(
            lift,
            speed=speed,
            joint_tolerance=joint_tolerance,
            cartesian_tolerance=cartesian_tolerance,
            label='hook_probe_lift',
        )
        probe_delta = probe[:2] - pick[:2]
        outward = np.hstack((
            [
                tip_position[0] + probe_delta[0],
                tip_position[1] + probe_delta[1],
                probe_z,
            ],
            rotation,
        ))
        success &= self.movep_precise(
            outward,
            speed=speed,
            joint_tolerance=joint_tolerance,
            cartesian_tolerance=cartesian_tolerance,
            label='hook_probe_out',
        )
        if self._ccda_motion_events[-1]['achieved_fraction'] < min_achieved_fraction:
            success = False
        if hold_steps:
            self.step_physics(int(hold_steps))

        return_delta = returned[:2] - pick[:2]
        return_pose = np.hstack((
            [
                tip_position[0] + return_delta[0],
                tip_position[1] + return_delta[1],
                probe_z,
            ],
            rotation,
        ))
        success &= self.movep_precise(
            return_pose,
            speed=speed,
            joint_tolerance=joint_tolerance,
            cartesian_tolerance=cartesian_tolerance,
            label='hook_probe_return',
        )
        if return_hold_steps:
            self.step_physics(int(return_hold_steps))

        release_pose = return_pose.copy()
        release_pose[2] = float(tip_position[2])
        success &= self.movep_precise(
            release_pose,
            speed=speed,
            joint_tolerance=joint_tolerance,
            cartesian_tolerance=cartesian_tolerance,
            label='hook_probe_lower_release',
        )
        self.ee.release()
        if post_release_steps:
            self.step_physics(int(post_release_steps))

        retreat = release_pose.copy()
        retreat[2] = float(retreat_z)
        success &= self.movep(retreat, speed=speed)
        return bool(success)

    def pick_precise_latch_probe(
            self,
            pose0,
            lift_height=0.004,
            hold_steps=60,
            return_hold_steps=60,
            post_release_steps=120,
            approach_height=0.02,
            retreat_z=0.3,
            joint_tolerance=1e-4,
            cartesian_tolerance=2e-4,
            min_achieved_fraction=0.8):
        if not self.deterministic:
            raise RuntimeError('pick_precise_latch_probe requires fixed-step mode')
        if lift_height <= 0:
            raise ValueError('lift_height must be positive')
        if min(hold_steps, return_hold_steps, post_release_steps) < 0:
            raise ValueError('hold steps must be non-negative')

        primitive = 'pick_precise_latch_probe'
        speed = 0.001
        delta_z = -0.0005
        if hasattr(self.task, 'primitive_params'):
            params = self.task.primitive_params[self.task.task_stage]
            speed = float(params.get('speed', speed))
            delta_z = float(params.get('delta_z', delta_z))
        if delta_z >= 0:
            raise ValueError('delta_z must be negative')

        deformable_ids = getattr(self.task, 'def_IDs', [])
        pick = np.asarray(pose0[0], dtype=np.float64)
        rotation = np.asarray(pose0[1], dtype=np.float64)
        success = True
        approach = np.hstack((
            [pick[0], pick[1], pick[2] + approach_height], rotation
        ))
        success &= self.movep(approach, speed=speed)

        lower = approach.copy()
        floor_limit = max(0.0, float(pick[2]) - 0.01)
        while not self.ee.detect_contact(deformable_ids) and lower[2] > floor_limit:
            lower[2] += delta_z
            success &= self.movep(
                lower, speed=speed, joint_tolerance=joint_tolerance
            )
            if not success:
                return False

        self.ee.activate(self.objects, deformable_ids)
        if not self.ee.check_grasp():
            self.ee.release()
            return False

        grasp_tip = np.asarray(
            p.getLinkState(
                self.ur5, self.ee_tip_link, computeForwardKinematics=True
            )[0],
            dtype=np.float64,
        )
        lift = np.hstack((
            [grasp_tip[0], grasp_tip[1], grasp_tip[2] + float(lift_height)],
            rotation,
        ))
        success &= self.movep_precise(
            lift,
            speed=speed,
            joint_tolerance=joint_tolerance,
            cartesian_tolerance=cartesian_tolerance,
            label='latch_probe_lift',
            primitive=primitive,
        )
        if self._ccda_motion_events[-1]['achieved_fraction'] < min_achieved_fraction:
            success = False

        counter = getattr(self.task, 'physics_step_count', None)
        if not callable(counter):
            raise RuntimeError('latch probe task has no physics step counter')
        hold_start = int(counter())
        if hold_steps:
            self.step_physics(int(hold_steps))
        self._record_ccda_hold_event(
            primitive=primitive,
            stage='latch_probe_hold',
            physics_step_start=hold_start,
            physics_step_end=int(counter()),
        )

        release_pose = np.hstack((
            [grasp_tip[0], grasp_tip[1], grasp_tip[2]], rotation
        ))
        success &= self.movep_precise(
            release_pose,
            speed=speed,
            joint_tolerance=joint_tolerance,
            cartesian_tolerance=cartesian_tolerance,
            label='latch_probe_lower_release',
            primitive=primitive,
        )
        return_start = int(counter())
        if return_hold_steps:
            self.step_physics(int(return_hold_steps))
        self._record_ccda_hold_event(
            primitive=primitive,
            stage='latch_probe_return_hold',
            physics_step_start=return_start,
            physics_step_end=int(counter()),
        )

        self.ee.release()
        post_start = int(counter())
        if post_release_steps:
            self.step_physics(int(post_release_steps))
        self._record_ccda_hold_event(
            primitive=primitive,
            stage='latch_probe_post_release',
            physics_step_start=post_start,
            physics_step_end=int(counter()),
        )

        retreat = release_pose.copy()
        retreat[2] = float(retreat_z)
        success &= self.movep(retreat, speed=speed)
        return bool(success)

    def pick_precise_tension_extension(
            self,
            pose0,
            pose_stage1,
            pose1,
            lift_height=0.004,
            approach_height=0.02,
            retreat_z=0.3,
            joint_tolerance=1e-4,
            cartesian_tolerance=2e-4,
            min_achieved_fraction=0.8,
            acquisition_motion_mode='legacy_joint_return'):
        """One grasp, two collinear waypoints, then release."""
        if not self.deterministic:
            raise RuntimeError(
                'pick_precise_tension_extension requires fixed-step mode'
            )

        lift_height = float(lift_height)
        approach_height = float(approach_height)
        retreat_z = float(retreat_z)
        joint_tolerance = float(joint_tolerance)
        cartesian_tolerance = float(cartesian_tolerance)
        min_achieved_fraction = float(min_achieved_fraction)
        acquisition_motion_mode = str(
            acquisition_motion_mode
        )
        if acquisition_motion_mode not in {
                'legacy_joint_return',
                'precise_endpoint_recovery'}:
            raise ValueError(
                'unsupported acquisition_motion_mode '
                f'{acquisition_motion_mode!r}'
            )
        precise_acquisition = (
            acquisition_motion_mode
            == 'precise_endpoint_recovery'
        )
        if lift_height <= 0:
            raise ValueError('lift_height must be positive')
        if approach_height <= lift_height:
            raise ValueError('approach_height must exceed lift_height')
        if retreat_z <= 0:
            raise ValueError('retreat_z must be positive')
        if joint_tolerance <= 0:
            raise ValueError('joint_tolerance must be positive')
        if cartesian_tolerance <= 0:
            raise ValueError('cartesian_tolerance must be positive')
        if not 0 < min_achieved_fraction <= 1:
            raise ValueError('min_achieved_fraction must be in (0,1]')

        primitive = 'pick_precise_tension_extension'
        speed = 0.001
        delta_z = -0.0005
        if hasattr(self.task, 'primitive_params'):
            params = self.task.primitive_params[self.task.task_stage]
            speed = float(params.get('speed', speed))
            delta_z = float(params.get('delta_z', delta_z))
        if delta_z >= 0:
            raise ValueError('tension extension delta_z must be negative')

        counter = getattr(
            self.task,
            'physics_step_count',
            None,
        )
        if not callable(counter) and precise_acquisition:
            raise RuntimeError(
                'tension task has no physics step counter'
            )
        if not callable(counter):
            counter = lambda: 0
        acquisition_step_start = int(
            counter()
        )

        def record_acquisition(
                *,
                success,
                failure_reason,
                approach_success,
                contact_detected,
                grasp_active,
                constraint_available,
                lower_step_count):
            physics_step_end = int(
                counter()
            )
            event = {
                'primitive': primitive,
                'stage': 'tension_pull_acquisition',
                'label': 'tension_pull_acquisition',
                'success': bool(success),
                'failure_reason': failure_reason,
                'approach_success': bool(
                    approach_success
                ),
                'contact_detected': bool(
                    contact_detected
                ),
                'grasp_active_after': bool(
                    grasp_active
                ),
                'constraint_available_after': bool(
                    constraint_available
                ),
                'lower_step_count': int(
                    lower_step_count
                ),
                'physics_step_start': (
                    acquisition_step_start
                ),
                'physics_step_end': (
                    physics_step_end
                ),
                'physics_step_count': int(
                    physics_step_end
                    - acquisition_step_start
                ),
                'achieved_fraction': (
                    1.0 if success else 0.0
                ),
                'joint_motion_success': bool(
                    approach_success
                    and failure_reason
                    not in {
                        'approach_motion_failed',
                        'contact_lowering_failed',
                    }
                ),
                'joint_timeout_count': 0,
                'timeout_reason': (
                    None
                    if success
                    else failure_reason
                ),
            }
            if precise_acquisition:
                self._ccda_motion_events.append(
                    event
                )
            return event

        deformable_ids = getattr(self.task, 'def_IDs', [])
        pick = np.asarray(pose0[0], dtype=np.float64)
        stage1 = np.asarray(pose_stage1[0], dtype=np.float64)
        final = np.asarray(pose1[0], dtype=np.float64)
        rotation = np.asarray(pose0[1], dtype=np.float64)
        for name, value in (
                ('pick', pick), ('stage1', stage1), ('final', final)):
            if value.shape != (3,):
                raise ValueError('{} position must be XYZ'.format(name))
            if not np.all(np.isfinite(value)):
                raise ValueError('{} position is non-finite'.format(name))

        stage1_vector = stage1[:2] - pick[:2]
        final_vector = final[:2] - pick[:2]
        stage1_distance = float(np.linalg.norm(stage1_vector))
        final_distance = float(np.linalg.norm(final_vector))
        if stage1_distance <= 1e-12:
            raise ValueError('stage-1 displacement is zero')
        if final_distance <= stage1_distance:
            raise ValueError('final displacement must exceed stage 1')
        stage1_direction = stage1_vector / stage1_distance
        final_direction = final_vector / final_distance
        if float(np.linalg.norm(stage1_direction - final_direction)) > 1e-9:
            raise ValueError('stage-1 and final pulls must be collinear')

        success = True
        approach = np.hstack((
            [pick[0], pick[1], pick[2] + approach_height], rotation
        ))

        if precise_acquisition:
            approach_success = (
                self.movep_precise(
                    approach,
                    speed=speed,
                    joint_tolerance=joint_tolerance,
                    cartesian_tolerance=(
                        cartesian_tolerance
                    ),
                    label='tension_pull_approach',
                    primitive=primitive,
                )
            )
        else:
            approach_success = self.movep(
                approach,
                speed=speed,
            )

        success &= bool(approach_success)
        if not approach_success:
            record_acquisition(
                success=False,
                failure_reason=(
                    'approach_motion_failed'
                ),
                approach_success=False,
                contact_detected=False,
                grasp_active=False,
                constraint_available=False,
                lower_step_count=0,
            )
            return False

        lower = approach.copy()
        floor_limit = max(
            0.0,
            float(pick[2]) - 0.01,
        )
        lower_step_count = 0

        while (
                not self.ee.detect_contact(
                    deformable_ids
                )
                and lower[2] > floor_limit):
            lower[2] = max(
                floor_limit,
                float(lower[2] + delta_z),
            )
            if precise_acquisition:
                lower_success = (
                    self.movep_precise(
                        lower,
                        speed=speed,
                        joint_tolerance=(
                            joint_tolerance
                        ),
                        cartesian_tolerance=(
                            cartesian_tolerance
                        ),
                        label=(
                            'tension_pull_lower_step'
                        ),
                        primitive=primitive,
                        record_event=False,
                    )
                )
            else:
                lower_success = self.movep(
                    lower,
                    speed=speed,
                    joint_tolerance=(
                        joint_tolerance
                    ),
                )

            lower_step_count += 1
            success &= bool(lower_success)
            if not lower_success:
                record_acquisition(
                    success=False,
                    failure_reason=(
                        'contact_lowering_failed'
                    ),
                    approach_success=True,
                    contact_detected=False,
                    grasp_active=False,
                    constraint_available=False,
                    lower_step_count=(
                        lower_step_count
                    ),
                )
                return False

        contact_detected = bool(
            self.ee.detect_contact(
                deformable_ids
            )
        )
        if not contact_detected:
            record_acquisition(
                success=False,
                failure_reason=(
                    'contact_not_detected'
                ),
                approach_success=True,
                contact_detected=False,
                grasp_active=False,
                constraint_available=False,
                lower_step_count=(
                    lower_step_count
                ),
            )
            return False

        self.ee.activate(
            self.objects,
            deformable_ids,
        )
        grasp_active = bool(
            self.ee.check_grasp()
        )
        constraint_available = bool(
            getattr(
                self.ee,
                'contact_constraint',
                None,
            )
            is not None
        )

        if not grasp_active:
            record_acquisition(
                success=False,
                failure_reason='grasp_failed',
                approach_success=True,
                contact_detected=True,
                grasp_active=False,
                constraint_available=(
                    constraint_available
                ),
                lower_step_count=(
                    lower_step_count
                ),
            )
            self.ee.release()
            retreat = approach.copy()
            retreat[2] = retreat_z
            self.movep(retreat, speed=speed)
            return False

        record_acquisition(
            success=True,
            failure_reason=None,
            approach_success=True,
            contact_detected=True,
            grasp_active=True,
            constraint_available=(
                constraint_available
            ),
            lower_step_count=(
                lower_step_count
            ),
        )

        grasp_tip = np.asarray(
            p.getLinkState(
                self.ur5, self.ee_tip_link, computeForwardKinematics=True
            )[0],
            dtype=np.float64,
        )
        pull_z = float(grasp_tip[2] + lift_height)

        def precise_move(target, label):
            nonlocal success
            success &= self.movep_precise(
                target,
                speed=speed,
                joint_tolerance=joint_tolerance,
                cartesian_tolerance=cartesian_tolerance,
                label=label,
                primitive=primitive,
            )
            event = self._ccda_motion_events[-1]
            grasp_active = bool(self.ee.check_grasp())
            event['grasp_active_after'] = grasp_active
            event['constraint_available_after'] = bool(
                getattr(self.ee, 'contact_constraint', None) is not None
            )
            if float(event['achieved_fraction']) < min_achieved_fraction:
                success = False
            if not grasp_active:
                success = False
            return grasp_active

        lift = np.hstack((
            [grasp_tip[0], grasp_tip[1], pull_z], rotation
        ))
        if not precise_move(lift, 'tension_pull_lift'):
            self.ee.release()
            return False

        stage1_delta = stage1[:2] - pick[:2]
        stage1_pose = np.hstack((
            [
                grasp_tip[0] + stage1_delta[0],
                grasp_tip[1] + stage1_delta[1],
                pull_z,
            ],
            rotation,
        ))
        if not precise_move(stage1_pose, 'tension_pull_stage1'):
            self.ee.release()
            return False

        final_delta = final[:2] - pick[:2]
        final_pose = np.hstack((
            [
                grasp_tip[0] + final_delta[0],
                grasp_tip[1] + final_delta[1],
                pull_z,
            ],
            rotation,
        ))
        if not precise_move(final_pose, 'tension_pull_stage2'):
            self.ee.release()
            return False

        release_pose = final_pose.copy()
        release_pose[2] = float(grasp_tip[2])
        success &= self.movep_precise(
            release_pose,
            speed=speed,
            joint_tolerance=joint_tolerance,
            cartesian_tolerance=cartesian_tolerance,
            label='tension_pull_lower_release',
            primitive=primitive,
        )
        release_event = self._ccda_motion_events[-1]
        release_event['grasp_active_before_release'] = bool(
            self.ee.check_grasp()
        )
        if float(release_event['achieved_fraction']) < min_achieved_fraction:
            success = False

        self.ee.release()
        retreat = release_pose.copy()
        retreat[2] = retreat_z
        success &= self.movep(retreat, speed=speed)
        return bool(success)

    def sweep(self, pose0, pose1):
        """Execute sweeping primitive."""
        success = True
        position0 = np.float32(pose0[0])
        position1 = np.float32(pose1[0])
        direction = position1 - position0
        length = np.linalg.norm(position1 - position0)
        if length == 0:
            direction = np.float32([0, 0, 0])
        else:
            direction = (position1 - position0) / length

        theta = np.arctan2(direction[1], direction[0])
        rotation = p.getQuaternionFromEuler((0, 0, theta))

        over0 = position0.copy()
        over0[2] = 0.3
        over1 = position1.copy()
        over1[2] = 0.3

        success &= self.movep(np.hstack((over0, rotation)))
        success &= self.movep(np.hstack((position0, rotation)))

        num_pushes = np.int32(np.floor(length / 0.01))
        for i in range(num_pushes):
            target = position0 + direction * num_pushes * 0.01
            success &= self.movep(np.hstack((target, rotation)), speed=0.003)

        success &= self.movep(np.hstack((position1, rotation)), speed=0.003)
        success &= self.movep(np.hstack((over1, rotation)))
        return success

    def push(self, pose0, pose1):
        """Execute pushing primitive."""
        p0 = np.float32(pose0[0])
        p1 = np.float32(pose1[0])
        p0[2], p1[2] = 0.025, 0.025
        if np.sum(p1 - p0) == 0:
            push_direction = 0
        else:
            push_direction = (p1 - p0) / np.linalg.norm((p1 - p0))
        p1 = p0 + push_direction * 0.01
        success &= self.movep(np.hstack((p0, self.home_pose[3:])))
        success &= self.movep(np.hstack((p1, self.home_pose[3:])), speed=0.003)
        return success

    #-------------------------------------------------------------------------
    # Motion Primitives
    #-------------------------------------------------------------------------

    def is_softbody_env(self):
        """In addition to this, please check task.py. In particular...

        Check the (a) policy's action, (b) reward, (c) done conditions.
        """
        return self.is_cloth_env() or self.is_bag_env()

    def is_new_cable_env(self):
        """I want a way to track new cable-related stuff alone."""
        return (isinstance(self.task, tasks.names['cable-shape']) or
                isinstance(self.task, tasks.names['cable-shape-notarget']) or
                isinstance(self.task, tasks.names['cable-line-notarget']) or
                isinstance(self.task, tasks.names['ccda-slack-cable-v2']) or
                isinstance(self.task, tasks.names['ccda-hidden-friction-cable']) or
                isinstance(self.task, tasks.names['cable-ring']) or
                isinstance(self.task, tasks.names['cable-ring-notarget']))

    def is_cloth_env(self):
        """Keep this updated when I adjust environment names."""
        return (isinstance(self.task, tasks.names['cloth-flat']) or
                isinstance(self.task, tasks.names['cloth-flat-notarget']) or
                isinstance(self.task, tasks.names['cloth-cover']))

    def is_bag_env(self):
        """Keep this updated when I adjust environment names."""
        return (isinstance(self.task, tasks.names['bag-alone-open']) or
                isinstance(self.task, tasks.names['bag-items-easy']) or
                isinstance(self.task, tasks.names['bag-items-hard']) or
                isinstance(self.task, tasks.names['bag-color-goal']))
