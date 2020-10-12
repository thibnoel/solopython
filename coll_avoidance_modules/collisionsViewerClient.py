from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import c_double
import matplotlib.pyplot as plt
import pinocchio as pin
import numpy as np
import time

# Helper functions
def visualizeCollisionDist(gv, p1, p2, name, color, init=False):
    ### --- display witness as normal patch tangent to capsule
    if(init):
        for i in range(2):
                gv.addCylinder('world/pinocchio/collisions/simple_patch_' + name + '_%d'%i, .01, .003, color)
        gv.addLine('world/pinocchio/collisions/line_' + name, p1.tolist(), p2.tolist(), color)

    direc = (p2-p1)/np.linalg.norm(p2-p1) 

    M1 = pin.SE3(pin.Quaternion.FromTwoVectors(np.matrix([0,0,1]).T,p1-p2).matrix(),p1)
    M2 = pin.SE3(pin.Quaternion.FromTwoVectors(np.matrix([0,0,1]).T,p2-p1).matrix(),p2)
    gv.applyConfiguration('world/pinocchio/collisions/simple_patch_' + name + '_0',pin.SE3ToXYZQUATtuple(M1))
    gv.applyConfiguration('world/pinocchio/collisions/simple_patch_' + name + '_1',pin.SE3ToXYZQUATtuple(M2))
    gv.setLineExtremalPoints('world/pinocchio/collisions/line_' + name, p1.tolist(), p2.tolist())

    gv.setColor('world/pinocchio/collisions/simple_patch_' + name + '_0', color)
    gv.setColor('world/pinocchio/collisions/simple_patch_' + name + '_1', color)
    gv.setColor('world/pinocchio/collisions/line_' + name, color)

    gv.refresh()


def visualizePair(gv, rmodel, rdata, q, caps_frames, local_wpoints, color, world_frame=False, init=False):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)

    p0 = np.array(local_wpoints[0])
    p1 = np.array(local_wpoints[1])

    p0.resize(3,1)
    p1.resize(3,1)
    
    leg_seg0 = rmodel.getFrameId(caps_frames[0])
    leg_seg1 = rmodel.getFrameId(caps_frames[1])
    if(not world_frame):
        p0 = rdata.oMf[leg_seg0].rotation@p0 + rdata.oMf[leg_seg0].translation
        p1 = rdata.oMf[leg_seg1].rotation@p1 + rdata.oMf[leg_seg1].translation
    else:
        p0 = local_wpoints[0]
        p1 = local_wpoints[1]
    
    p0.resize(3)
    p1.resize(3)
    
    visualizeCollisionDist(gv, np.array(p0), np.array(p1), caps_frames[0] + '_' + caps_frames[1], color, init=init)


def visualizeCollisions(gv, rmodel, rdata, q, caps_frames_list, legs_dist_list, wpoints_list, viz_thresh, legs_activation_thresh, init=False):
    for i in range(len(caps_frames_list)):
        color = [0,0,0,0]
        if(legs_dist_list[i]) < viz_thresh:
            color = [0,1,0,1] if legs_dist_list[i] > legs_activation_thresh else [1,0,0,1]
        visualizePair(gv, rmodel, rdata, q, caps_frames_list[i], wpoints_list[i], color, world_frame=False, init=init)


def visualizeTorques(gv, rmodel, rdata, tau_q, init=False):
    solo12 = (len(tau_q) == 12)
    for k in range(len(tau_q)):
        jointFrame = rdata.oMi[k+2]
        #jointFrame = rdata.oMi[k]
        name = 'world/pinocchio/collisions/torque_' + str(k)
        color = [0,0,1,1]

        additional_transl = np.array([0,0,0.0])

        if solo12:
            if(k%3==0):
                direc = [1,0,0] if tau_q[k]>0 else [-1,0,0]
                additional_transl = np.array([0,0,0.05])
            else:
                direc = [0,1,0] if tau_q[k]>0 else [0,-1,0]
        else:
            direc = [0,1,0] if tau_q[k]>0 else [0,-1,0]

        additional_transl.resize(3,1)
        orientation = pin.SE3(pin.Quaternion.FromTwoVectors(np.matrix([1,0,0]).T,np.matrix(direc).T).matrix(),jointFrame.translation + additional_transl)

        if(init):
            gv.addArrow(name, 0.003, 1, color)
        gv.resizeArrow(name, 0.003, np.abs(tau_q[k]))
        gv.applyConfiguration(name, pin.SE3ToXYZQUATtuple(orientation))


def visualizeShoulderDist(q_shoulder, dist, shd_thresh):
    plt.axis([-np.pi, np.pi, -np.pi, np.pi])
    if dist < shd_thresh:
        color = 'r'
    else:
        color = 'limegreen'
    out = plt.scatter(q_shoulder[0][0], q_shoulder[1][0], c=color, alpha=1)
    
    return out

def visualizeShoulderTorque(q_shoulder, dist, shd_thresh, shd_torque, scale=1.):
    x_vals = [q_shoulder[0][0], q_shoulder[0][0] + scale*shd_torque[0]]
    y_vals = [q_shoulder[1][0], q_shoulder[1][0] + scale*shd_torque[1]]
    
    if dist < shd_thresh:
        t = plt.plot(x_vals, y_vals, c='b', linestyle='-')
    else:
        t = plt.plot(x_vals, y_vals, c='b', linestyle='-', alpha=0)
    return t


class NonBlockingViewerFromRobot():
    def __init__(self,robot,dt=0.01,nb_pairs=0, viz_thresh=0, act_thresh_legs=0, act_thresh_shd=0, viz_shoulder=False):
        # a shared c_double array
        self.dt = dt
        self.nb_pairs = nb_pairs
        self.viz_shoulder = viz_shoulder
        self.shared_q_viewer = Array(c_double, robot.nq, lock=False)
        self.shared_tau = Array(c_double, robot.nq - 7, lock=False)
        self.shared_tau_shd = Array(c_double, robot.nq - 7, lock=False)
        self.shared_legs_dist = Array(c_double, nb_pairs, lock=False)
        self.shared_shd_dist = Array(c_double, 4, lock=False)
        self.shared_wpoints = []
        for k in range(self.nb_pairs):
            self.shared_wpoints.append([Array(c_double, 3), Array(c_double, 3)])
        
        self.p = Process(target=self.display_process, args=(robot, self.shared_q_viewer, self.shared_wpoints, self.shared_legs_dist, self.shared_shd_dist, self.shared_tau, self.shared_tau_shd, viz_thresh, act_thresh_legs, act_thresh_shd, self.viz_shoulder))
        self.p.start()
        
    def display_process(self,robot, shared_q_viewer, shared_wpoints, shared_legs_dist, shared_shd_dist, shared_tau, shared_tau_shd, viz_thresh, legs_activation_thresh, shd_activation_thresh, displayShoulder=False):
        robot.displayVisuals(True)
        robot.displayCollisions(True)
        ''' This will run on a different process'''
        q_viewer = robot.q0.copy()
        tau_q = np.zeros(robot.nq - 7)
        tau_q_shd = np.zeros(robot.nq - 7)
        legs_dist = np.zeros(self.nb_pairs)
        shd_dist = np.zeros(4)
        wpoints = [[[0,0,0],[0,0,0]]]*self.nb_pairs
        gv = robot.viewer.gui
        rmodel = robot.model
        rdata = rmodel.createData()

        if(len(tau_q)==8):
            caps_frames_list = [["FL_UPPER_LEG", "HL_LOWER_LEG"],\
                                ["FL_LOWER_LEG", "HL_UPPER_LEG"],
                                ["FL_LOWER_LEG", "HL_LOWER_LEG"],

                                ["FR_UPPER_LEG", "HR_LOWER_LEG"],
                                ["FR_LOWER_LEG", "HR_UPPER_LEG"],
                                ["FR_LOWER_LEG", "HR_LOWER_LEG"]]
        else:
            caps_frames_list = [["FL_UPPER_LEG", "FR_UPPER_LEG"],\
                                ["FL_UPPER_LEG", "FR_LOWER_LEG"],
                                ["FL_LOWER_LEG", "FR_UPPER_LEG"],
                                ["FL_LOWER_LEG", "FR_LOWER_LEG"],
                                
                                ["FL_UPPER_LEG", "HL_LOWER_LEG"],
                                ["FL_LOWER_LEG", "HL_UPPER_LEG"],
                                ["FL_LOWER_LEG", "HL_LOWER_LEG"],

                                ["FL_UPPER_LEG", "HR_LOWER_LEG"],
                                ["FL_LOWER_LEG", "HR_UPPER_LEG"],
                                ["FL_LOWER_LEG", "HR_LOWER_LEG"],
                                
                                ["FR_UPPER_LEG", "HL_LOWER_LEG"],
                                ["FR_LOWER_LEG", "HL_UPPER_LEG"],
                                ["FR_LOWER_LEG", "HL_LOWER_LEG"],
                                
                                ["FR_UPPER_LEG", "HR_LOWER_LEG"],
                                ["FR_LOWER_LEG", "HR_UPPER_LEG"],
                                ["FR_LOWER_LEG", "HR_LOWER_LEG"],
                                
                                ["HL_UPPER_LEG", "HR_UPPER_LEG"],
                                ["HL_UPPER_LEG", "HR_LOWER_LEG"],
                                ["HL_LOWER_LEG", "HR_UPPER_LEG"],
                                ["HL_LOWER_LEG", "HR_LOWER_LEG"]]

        count = 0

        plots = [[]]*4
        line_plots = [[]]*4
        plt.figure()
        shd_dist_landscape = np.load('/home/tnoel/stage/solo-collisions/src/python/ref_net_dist_landscape.npy', allow_pickle=True)
        plt.suptitle("Shoulders distances")

        shd_dist_landscape = 1*(shd_dist_landscape > 0) + 1*(shd_dist_landscape > shd_activation_thresh) 

        for k in range(4):
            plt.subplot(2,2,k+1)
            plt.imshow(shd_dist_landscape, extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=plt.cm.gray)
        #plt.show()
        while(1):
            for n in gv.getNodeList():
                if 'LEG_0' in n and 'collision' in n and len(n)>27:
                    gv.setColor(n, [1,0.5,0,0.1])
            
            for i in range(robot.nq):
                q_viewer[i] = shared_q_viewer[i]
            
            for i in range(robot.nq - 7):
                tau_q[i] = shared_tau[i]

            for i in range(self.nb_pairs):
                wpoints[i] = shared_wpoints[i]
                legs_dist[i] = shared_legs_dist[i]

            for i in range(4):
                shd_dist[i] = shared_shd_dist[i]

            robot.display(q_viewer)
            #print(q_viewer)
            #print(shd_dist[0])

            shoulders_names = ['FL', 'FR', 'HL', 'HR']
            
            #print(plots)
            for k in range(4):

                plt.subplot(2,2,k+1)
                #plt.imshow(shd_dist_landscape, extent=[-np.pi, np.pi, -np.pi, np.pi])
                plt.title(shoulders_names[k] + '\nd = {:.3f}'.format(shd_dist[k]))
                shd_torque = shared_tau_shd[3*k:3*k+2]
                plots[k].append(visualizeShoulderDist(q_viewer[7+3*k:7+3*k+2].tolist(), shd_dist[k], shd_activation_thresh))
                torque_line, = visualizeShoulderTorque(q_viewer[7+3*k:7+3*k+2].tolist(), shd_dist[k], shd_activation_thresh, shd_torque)
                line_plots[k].append(torque_line)
                #print(plots)
                if (len(plots[k]) > 4):
                    plots[k].pop(0).remove()
                if (len(line_plots[k]) > 4):
                    line_plots[k].pop(0).remove()

            plt.pause(self.dt)

            visualizeCollisions(gv, rmodel, rdata, q_viewer, caps_frames_list, legs_dist, wpoints, viz_thresh, legs_activation_thresh, init=(count==0))
            visualizeTorques(gv, rmodel, rdata, tau_q, init=(count==0))

            gv.refresh()
            time.sleep(self.dt)
            count += 1
        
        plt.show()

    def display(self,q):
        for i in range(len(self.shared_q_viewer)):
            self.shared_q_viewer[i] = q[i]

    def display_tau(self, tau):
        for i in range(len(self.shared_tau)):
            self.shared_tau[i] = tau[i]
    
    def display_tau_shd(self, tau_shd):
        for i in range(len(self.shared_tau_shd)):
            self.shared_tau_shd[i] = tau_shd[i]

    def display_legs_dist(self, legs_dist):
        for i  in range(len(self.shared_legs_dist)):
            self.shared_legs_dist[i] = legs_dist[i]

    def display_shd_dist(self, shd_dist):
        for i  in range(len(self.shared_shd_dist)):
            self.shared_shd_dist[i] = shd_dist[i]

    def updateWitnessPoints(self, wpoints):
        for i in range(len(wpoints)):
            for j in range(3):
                self.shared_wpoints[i][0][j] = wpoints[i][0][j]
                self.shared_wpoints[i][1][j] = wpoints[i][1][j]

    def stop(self):
        self.p.terminate()
        self.p.join()


class viewerClient():
    #def __init__(self,urdf="/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots/solo_description/robots/solo.urdf",modelPath="/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots",dt=0.01):
    def __init__(self, legs_pairs_dist_threshold, shoulders_dist_threshold, urdf = "/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo.urdf", modelPath="/opt/openrobots/share/example-robot-data/robots/solo_description/robots/"):
        pin.switchToNumpyMatrix()
        robot = pin.RobotWrapper.BuildFromURDF( urdf, modelPath, pin.JointModelFreeFlyer())
        robot.initViewer(loadModel=True)   
        if ('viewer' in robot.viz.__dict__):
            robot.viewer.gui.setRefreshIsSynchronous(False)       

        dt = 0.01 

        self.nbv = NonBlockingViewerFromRobot(robot,dt, nb_pairs=20, viz_thresh=3*legs_pairs_dist_threshold, act_thresh_legs=legs_pairs_dist_threshold, act_thresh_shd=shoulders_dist_threshold, viz_shoulder=True)

    def display(self,q, legs_dist, shd_dist, wpoints, tau, tau_shd):
        self.nbv.display(q)
        self.nbv.display_tau(tau)
        self.nbv.display_tau_shd(tau_shd)
        self.nbv.display_legs_dist(legs_dist)
        self.nbv.display_shd_dist(shd_dist)
        self.nbv.updateWitnessPoints(wpoints)

    def stop(self):
        self.nbv.stop()

"""v=viewerClient()
from IPython import embed
embed()"""

