# coding: utf8
from coll_avoidance_modules.solo_coll_wrapper_c import *
from coll_avoidance_modules.collisions_controller import *

from utils.logger import Logger

import numpy as np
import argparse
import math
from time import clock, sleep
from solo8 import Solo8


def example_script(name_interface, clib_path):
	device = Solo8(name_interface,dt=0.001)
	nb_motors = device.nb_motors
	LOGGING = False
	
	qc = None
	if LOGGING:
		# Initialize logger
		qc = QualisysClient(ip="140.93.16.160", body_id=0) # ??
		logger = Logger(device, qualisys=qc, logSize=50000)
	
	#### Set collision avoidance parameters
	collision_threshold = 0.05
	collision_kp = 50.
	collision_kv = 0.3

	k_friction = 0.1

	emergency_dist_thresh = collision_threshold/5
	emergency_tau_thresh = 3

	# Load the specified compiled C library
	cCollFun = CDLL(clib_path)
	# Initialize emergency behavior trigger var.
	emergencyFlag = False

	device.Init(calibrateEncoders=True)
	#CONTROL LOOP ***************************************************
	tau_q = np.zeros(nb_motors)
	while ((not device.hardware.IsTimeout()) and (clock() < 200)):
		device.UpdateMeasurment()


		# Check if the controller switched to emergency mode
		if(emergencyFlag):
			# Compute emergency behavior
			# Ex :
			tau_q = 0*computeEmergencyTorque(device.v_mes, collision_kv)
		else:
			# Compute collisions distances and jacobians from the C lib. 
			c_results = getLegsCollisionsResults(device.q_mes, cCollFun, nb_motors, 6)
			c_dist_legs = getLegsDistances(c_results, nb_motors, 6)
			c_Jlegs = getLegsJacobians(c_results, nb_motors, 6)
			# Compute collision avoidance torque
			tau_q = computeRepulsiveTorque(device.q_mes, device.v_mes, c_dist_legs, c_Jlegs, dist_thresh=collision_threshold, kp=collision_kp, kv=collision_kv)

		# Set a virtual friction torque to avoid divergence
		#tau_q += -k_friction*device.v_mes

		# Set the computed torque as command
		device.SetDesiredJointTorque(tau_q)
		# Check the condition for triggering emergency behavior
		#emergencyFlag = emergencyFlag or emergencyCondition(c_dist_legs, device.v_mes, tau_q, emergency_dist_thresh, emergency_tau_thresh)
		# Call logger
		if LOGGING:
        	    logger.sample(device, qualisys=qc)
	
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

	parser.add_argument('-C',
		        '--clib',
		        required=True,
		        help='Path to the compiled C-generated library used for distance and jacobian evaluations, for instance "libcoll_legs8.so"')

	example_script(parser.parse_args().interface, parser.parse_args().clib)


if __name__ == "__main__":
	main()
