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


def compute_pd(q_desired, v_desired, KP, KD, device, active_dof=[]):
    if(len(active_dof) > 0):
        arr_KP = np.zeros(12)
        arr_KV = np.zeros(12)
        arr_KP[active_dof] = KP
        arr_KV[active_dof] = KV

	pos_error = q_desired - device.q_mes
	vel_error = v_desired - device.v_mes
	tau = KP * pos_error + KD * vel_error #+ KT * tau_desired
	#tau = np.maximum(np.minimum(tau, tau_max), -tau_max) 
	return tau


def put_on_the_floor(device, q_init):
	global key_pressed
	key_pressed = False
	Kp_pos = 3.
	Kd_pos = 0.01
	imax = 3.0
	pos = np.zeros(device.nb_motors)
	for motor in range(device.nb_motors):
		pos[motor] = q_init[device.motorToUrdf[motor]] * device.gearRatioSigned[motor]
	listener = keyboard.Listener(on_press=on_press)
	listener.start()
	print("Put the robot on the floor and press Enter")
	while not key_pressed:
		device.UpdateMeasurment()
		for motor in range(device.nb_motors):
			ref = Kp_pos*(pos[motor] - device.hardware.GetMotor(motor).GetPosition() - Kd_pos*device.hardware.GetMotor(motor).GetVelocity())
			ref = min(imax, max(-imax, ref))
			device.hardware.GetMotor(motor).SetCurrentReference(ref)
		device.SendCommand(WaitEndOfCycle=True)

	print("Start the motion.")


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
    
    #### Set ref. traj. PD parameters
    ref_traj_KP = 0
    ref_traj_KV = 0
    active_dof = [0,1,2,3,4,5,6,7,8,9,10,11]

    #### Set collision avoidance parameters
    legs_threshold = 0.05
    legs_kp = 20.
    legs_kv = 0.0
    nb_legs_pairs = 20

    #### Shoulder collision parameters
    shd_threshold = 0.2
    shd_kp = 3.
    shd_kv = 0.

    ### Emergency behavior switches
    q_bounds = [-4,4]
    vq_max = 20.0
    tau_q_max = 1.0

    # Load the specified compiled C library
    cCollFun = CDLL(legs_clib_path)
    nnCCollFun = CDLL(shd_clib_path)
    # Initialize emergency behavior trigger var.
    emergencyFlag = 0

    # Initialize viewer
    if VIEWER:
        viewer_coll = viewerClient(nb_legs_pairs, 3, legs_threshold, shd_threshold, urdf="/home/ada/git/tnoel/solopython/coll_avoidance_modules/urdf/solo12_simplified.urdf", modelPath="/home/ada/git/tnoel/solopython/coll_avoidance_modules/urdf")

    device.Init(calibrateEncoders=True, q_init=q_init)

    put_on_the_floor(device, q_init)
    #CONTROL LOOP ***************************************************
    tau_q = np.zeros(nb_motors)
    while ((not device.hardware.IsTimeout()) and (clock() < 120) and emergencyFlag==0):
        device.UpdateMeasurment()

        tau_q[:] = 0.

        # Compute collisions distances and jacobians from the C lib. 
        c_results = getLegsCollisionsResults(device.q_mes, cCollFun, nb_motors, nb_legs_pairs, witnessPoints=True)
        c_dist_legs = getLegsDistances(c_results, nb_motors, nb_legs_pairs, witnessPoints=True)
        c_Jlegs = getLegsJacobians(c_results, nb_motors, nb_legs_pairs, witnessPoints=True)
        c_wPoints = getLegsWitnessPoints(c_results, nb_motors, nb_legs_pairs)
        
        ### Get results from C generated code (shoulder neural net)
        #c_shd_dist, c_shd_jac = getAllShouldersCollisionsResults(device.q_mes, nnCCollFun, 2, offset=0.08) # 2D neural net
        c_shd_dist, c_shd_jac = getAllShouldersCollisionsResults(device.q_mes, nnCCollFun, 3, offset=0.11) #offset with 3 inputs: 0.18 (small), 0.11 (large)"

        # Compute collision avoidance torque
        tau_legs = computeRepulsiveTorque(device.q_mes, device.v_mes, c_dist_legs, c_Jlegs, legs_threshold, legs_kp, legs_kv, opposeJacIfNegDist=True)
        tau_shd = computeRepulsiveTorque(device.q_mes, device.v_mes, c_shd_dist, c_shd_jac, shd_threshold, shd_kp, shd_kv, opposeJacIfNegDist=False)

        tau_q = 1*tau_legs + 1*tau_shd

        # Set the computed torque as command
        device.SetDesiredJointTorque(tau_q)
        # Check the condition for triggering emergency behavior
        emergencyFlag = max(emergencyFlag, emergencyCondition(device.q_mes, device.v_mes, tau_q, q_bounds, vq_max, tau_q_max))
        # Call logger
        if LOGGING:
            logger.sample(device, qualisys=qc)

        if VIEWER :
            viewer_coll.display(np.concatenate(([0,0,0,0,0,0,0],device.q_mes)), c_dist_legs, c_shd_dist, c_wPoints, tau_legs, tau_shd)

        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 100) == 0):
            device.Print()
            print(tau_q)


        #****************************************************************

    print("Emergency :  {}".format(emergencyFlag))

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
