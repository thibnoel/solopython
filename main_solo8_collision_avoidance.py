# coding: utf8
from coll_avoidance_modules.solo_coll_wrapper_c import *
from coll_avoidance_modules.collisions_controller import *

import numpy as np
import argparse
import math
from time import clock, sleep
from utils.viewerClient import viewerClient
from solo8 import Solo8


def example_script(name_interface, clib_path):
    viewer = viewerClient()
    device = Solo8(name_interface,dt=0.001)
    nb_motors = device.nb_motors

    q_viewer = np.array((7 + nb_motors) * [0.,])

    #### Set collision avoidance parameters
    collision_threhsold = 0.1
    collision_kp = 1.
    collision_kv = 0.

    k_friction = 0.1

    emergency_dist_thresh = collision_threhsold/5
    emergency_tau_thresh = 3

    # Load the specified compiled C library
    cCollFun = CDLL(clib_path)
	# Initialize emergency behavior trigger var.
	emergencyFlag = False

    device.Init(calibrateEncoders=False)
    #CONTROL LOOP ***************************************************
    while ((not device.hardware.IsTimeout()) and (clock() < 200)):
        device.UpdateMeasurment()
		
        tau_q = np.zeros(nb_motors)
		# Check if the controller switched to emergency mode
		if(emergencyFlag):
			# Compute emergency behavior
			# Ex :
			tau_q = computeEmergencyTorque(device.v_mes, collision_kv)
		else:
		    # Compute collisions distances and jacobians from the C lib. 
			c_results = getLegsCollisionsResults(device.q_mes, cCollFun, nb_motors, 6)
			c_dist_legs = getLegsDistances(c_results, nb_motors, 6)
			c_Jlegs = getLegsJacobians(c_results, nb_motors, 6)
		    # Compute collision avoidance torque
            tau_q = computeRepulsiveTorque(device.q_mes, device.v_mes, c_dist_legs, c_Jlegs, dist_thresh=collision_threshold, kp=collision_kp, kv=collision_kv)

        # Set a virtual friction torque to avoid divergence
        tau_q += -k_friction*device.v_mes

        # Set the computed torque as command
        device.SetDesiredJointTorque(tau_q)
        # Check the condition for triggering emergency behavior
        emergencyFlag = emergencyFlag or emergencyCondition(c_dist_legs, device.v_mes, tau_q, emergency_dist_thresh, emergency_tau_thresh):

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
