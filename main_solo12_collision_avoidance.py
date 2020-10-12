# coding: utf8
from coll_avoidance_modules.solo_coll_wrapper_c import *
from coll_avoidance_modules.collisions_controller import *
from coll_avoidance_modules.collisionsViewerClient import *

from utils.logger import Logger

import numpy as np
import argparse
import math
from time import clock, sleep
from solo12 import Solo12


def example_script(name_interface, legs_clib_path, shd_clib_path):
    device = Solo12(name_interface,dt=0.001)
    nb_motors = device.nb_motors
    LOGGING = False
    VIEWER = True
    
    qc = None
    if LOGGING:
        # Initialize logger
        qc = QualisysClient(ip="140.93.16.160", body_id=0) # ??
        logger = Logger(device, qualisys=qc, logSize=50000)
    
    #### Set collision avoidance parameters
    legs_threshold = 0.05
    legs_kp = 75.
    legs_kv = 0.4

    nb_legs_pairs = 6

    emergency_dist_thresh = legs_threshold/5
    emergency_tau_thresh = 3

    #### Shoulder collision parameters
    shd_threshold = 0.5
    shd_kp = 5
    shd_kv = 0.

    # Load the specified compiled C library
    cCollFun = CDLL(legs_clib_path)
    nnCCollFun = CDLL(shd_clib_path)
    # Initialize emergency behavior trigger var.
    emergencyFlag = False

    # Initialize viewer
    if VIEWER:
        viewer_coll = viewerClient(legs_threshold, urdf="/home/ada/git/tnoel/solopython/coll_avoidance_modules/urdf/solo8_simplified.urdf", modelPath="/home/ada/git/tnoel/solopython/coll_avoidance_modules/urdf")

    device.Init(calibrateEncoders=True)
    #CONTROL LOOP ***************************************************
    tau_q = np.zeros(nb_motors)
    while ((not device.hardware.IsTimeout()) and (clock() < 200)):
        device.UpdateMeasurment()


        # Check if the controller switched to emergency mode
        if(emergencyFlag):
            # Compute emergency behavior
            # Ex :
            tau_q = 0*computeEmergencyTorque(device.v_mes, legs_kv)
        else:
            # Compute collisions distances and jacobians from the C lib. 
            c_results = getLegsCollisionsResults(device.q_mes, cCollFun, nb_motors, nb_legs_pairs, witnessPoints=True)
            c_dist_legs = getLegsDistances(c_results, nb_motors, nb_legs_pairs, witnessPoints=True)
            c_Jlegs = getLegsJacobians(c_results, nb_motors, nb_legs_pairs, witnessPoints=True)
            c_wPoints = getLegsWitnessPoints(c_results, nb_motors, nb_legs_pairs)
            
            ### Get results from C generated code (shoulder neural net)
            c_shd_dist, c_shd_jac = getAllShouldersCollisionsResults(q, nnCCollFun, 2, offset=0.08)

            # Compute collision avoidance torque
            tau_legs = computeRepulsiveTorque(device.q_mes, device.v_mes, c_dist_legs, c_Jlegs, dist_thresh=legs_threshold, kp=legs_kp, kv=legs_kv)
            tau_shd = computeRepulsiveTorque(device.q_mes, device.v_mes, c_shd_dist, c_shd_jac, dist_thresh=shd_threshold, kp=shd_kp, kv=shd_kv)

            tau_q += 0*tau_legs + 0*tau_shd

        # Set the computed torque as command
        device.SetDesiredJointTorque(tau_q)
        # Check the condition for triggering emergency behavior
        #emergencyFlag = emergencyFlag or emergencyCondition(c_dist_legs, device.v_mes, tau_q, emergency_dist_thresh, emergency_tau_thresh)
        # Call logger
        if LOGGING:
            logger.sample(device, qualisys=qc)

        if VIEWER :
            viewer_coll.display(np.concatenate(([0,0,0,0,0,0,0],q)), c_dist_legs, c_shd_dist, c_wPoints, tau_legs, tau_shd)
    
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 100) == 0):
            device.Print()
            print(tau_q)


        #****************************************************************

    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)
    
    # Save the logs of the Logger object
    if LOGGING:
        logger.saveAll()
        print("Log saved")
    
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

    parser.add_argument('-CL',
                '--cliblegs',
                required=True,
                help='Path to the compiled C-generated library used for distance and jacobian evaluations, for instance "libcoll_legs8.so"')

    parser.add_argument('-CS',
                '--clibshd',
                required=True,
                help='Path to the compiled C-generated library used for shoulder distance and jacobian evaluations, for instance "libcoll_nn.so"')

    example_script(parser.parse_args().interface, parser.parse_args().cliblegs, parser.parse_args().clibshd)


if __name__ == "__main__":
    main()
