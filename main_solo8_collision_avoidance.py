# coding: utf8
from coll_avoidance_modules.legs_cpp_wrapper import *

import numpy as np
import argparse
import math
from time import clock, sleep
from utils.viewerClient import viewerClient
from solo8 import Solo8

def compute_coll_avoidance_torque(q, vq, clib, dist_thresh=0.1, kp=0, kv=0, nb_motors=8, active_pairs=[]):
    # Initialize repulsive torque
    tau_avoid = np.zeros(nb_motors)
    
    # Compute collisions distances and jacobians from the C lib. 
    c_results = getLegsCollisionsResults8(q, clib)
    c_dist_legs = getDistances8(c_results)
    c_Jlegs = getJacobians8(c_results)
    
    if(len(active_pairs)==0):
        active_pairs = [i for i in range(len(c_dist_legs))]

    # Loop through the distance results to check for threshold violation
    for i in range(len(c_dist_legs)):
        J = c_Jlegs[i]
        d = c_dist_legs[i]
        
        tau_rep = np.zeros(nb_motors)
        # If violation, compute viscoelastic repulsive torque along the collision jacobian
        if(d < dist_thresh and (i in active_pairs)):
            tau_rep = -kp*(d - dist_thresh) - kv*J@vq
        tau_avoid += tau_rep*J.T
    
    return tau_avoid

def example_script(name_interface, clib_path):
    viewer = viewerClient()
    device = Solo8(name_interface,dt=0.001)
    nb_motors = device.nb_motors

    q_viewer = np.array((7 + nb_motors) * [0.,])

    #### Set collision avoidance parameters
    collision_threhsold = 0.1
    collision_kp = 1.
    collision_kv = 0.

    # Load the specified compiled C library
    cCollFun = CDLL(clib_path)

    device.Init(calibrateEncoders=False)
    #CONTROL LOOP ***************************************************
    while ((not device.hardware.IsTimeout()) and (clock() < 200)):
        device.UpdateMeasurment()
        
        #device.SetDesiredJointTorque([0]*nb_motors)
        # Compute collision avoidance torque
        tau_coll_avoid = compute_coll_avoidance_torque(device.q_mes, device.v_mes, cCollFun, dist_thresh=collision_threshold, kp=collision_kp, kd=collision_kv)
        device.SetDesiredJointTorque(tau_coll_avoid)

        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 100) == 0):
            device.Print()

        q_viewer[3:7] = device.baseOrientation  # IMU Attitude
        q_viewer[7:] = device.q_mes  # Encoders
        viewer.display(q_viewer)
    #****************************************************************
    
    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)

    if device.hardware.IsTimeout():
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")
    device.hardware.Stop()  # Shut down the interface between the computer and the master board

def main():
    parser = argparse.ArgumentParser(description='Example masterboard use in python.')
    parser.add_argument('-i',
                        '--interface',
                        required=True,
                        help='Name of the interface (use ifconfig in a terminal), for instance "enp1s0"')

    parser.add_argument('-C',
                        '--clib',
                        required=True,
                        help='Path to the compiled C-generated library used for distance and jacobian evaluations, for instance "libcoll_legs8.so"')

    example_script(parser.parse_args().interface, parser.parse_args().clib)


if __name__ == "__main__":
    main()
