# coding: utf8

##################################
	
# Pinocchio modules
import pinocchio as pin # Pinocchio library
from pinocchio.utils import * # Utilitary functions from Pinocchio
from pinocchio.robot_wrapper import RobotWrapper # Robot Wrapper to load an URDF in Pinocchio
	
##################################

# Other modules
import numpy as np
from PD import *


#####################
## FEET CONTROLLER ##
#####################

# Initialization of the controller's parameters
q_ref = np.zeros((15,1))
flag_q_ref = True

def c(q, qdot, dt, solo):
	
	qu = q[:7] # unactuated, [x, y, z] position of the base + [x, y, z, w] orientation of the base (stored as a quaternion)
	qa = q[7:] # actuated, [q1, q2, ..., q8] angular position of the 8 motors
	qu_dot = qdot[:6] # [v_x, v_y, v_z] linear velocity of the base and [w_x, w_y, w_z] angular velocity of the base along x, y, z axes of the world
	qa_dot = qdot[6:] # angular velocity of the 8 motors
	
	qa_ref = np.zeros((8,1)) # target angular positions for the motors
	qa_dot_ref = np.zeros((8,1)) # target angular velocities for the motors

	#################################################
	
	from numpy.linalg import pinv
	
	global q_ref, flag_q_ref
	
	if flag_q_ref:
		q_ref = solo.q0.copy()
		flag_q_ref = False
		
	# Initialization of the variables needed for the controller
	J_post = np.eye(8)		# jacobian of posture task
	omega = 10e-3			# weight of the posture task
	Kp = 100				# convergence gain
	# Frame index of each foot
	ID_FL = solo.model.getFrameId("FL_FOOT")
	ID_FR = solo.model.getFrameId("FR_FOOT")
	ID_HL = solo.model.getFrameId("HL_FOOT")
	ID_HR = solo.model.getFrameId("HR_FOOT")
	
	# compute/update all joints and frames
	pin.forwardKinematics(solo.model, solo.data, q_ref)
	
	# Getting the current height (on axis z) of each foot
	hFL = solo.data.oMf[ID_FL].translation[2]
	hFR = solo.data.oMf[ID_FR].translation[2]
	hHL = solo.data.oMf[ID_HL].translation[2]
	hHR = solo.data.oMf[ID_HR].translation[2]
	
	# Computing the error in the world frame
	err_FL = np.concatenate((np.zeros([2,1]),hFL))
	err_FR = np.concatenate((np.zeros([2,1]),hFR))
	err_HL = np.concatenate((np.zeros([2,1]),hHL))
	err_HR = np.concatenate((np.zeros([2,1]),hHR))
	
	# Error of posture
	err_post = q - q_ref
	
	# Computing the error in the local frame
	oR_FL = solo.data.oMf[ID_FL].rotation
	oR_FR = solo.data.oMf[ID_FR].rotation
	oR_HL = solo.data.oMf[ID_HL].rotation
	oR_HR = solo.data.oMf[ID_HR].rotation
	
	# Getting the different Jacobians
	fJ_FL3 = pin.frameJacobian(solo.model, solo.data, q_ref, ID_FL)[:3,-8:]    #Take only the translation terms
	oJ_FL3 = oR_FL*fJ_FL3    #Transformation from local frame to world frame
	oJ_FLz = oJ_FL3[2,-8:]    #Take the z_component
	
	fJ_FR3 = pin.frameJacobian(solo.model, solo.data, q_ref, ID_FR)[:3,-8:]
	oJ_FR3 = oR_FR*fJ_FR3
	oJ_FRz = oJ_FR3[2,-8:]
	
	fJ_HL3 = pin.frameJacobian(solo.model, solo.data, q_ref, ID_HL)[:3,-8:]
	oJ_HL3 = oR_HL*fJ_HL3
	oJ_HLz = oJ_HL3[2,-8:]
	
	fJ_HR3 = pin.frameJacobian(solo.model, solo.data, q_ref, ID_HR)[:3,-8:]
	oJ_HR3 = oR_HR*fJ_HR3
	oJ_HRz = oJ_HR3[2,-8:]
	
	# Displacement and posture error
	nu = np.vstack([err_FL[2],err_FR[2], err_HL[2], err_HR[2], omega*err_post[7:]])
	
	# Making a single z-row Jacobian vector plus the posture Jacobian
	J = np.vstack([oJ_FLz, oJ_FRz, oJ_HLz, oJ_HRz, omega*J_post])
	
	# Computing the velocity
	qa_dot_ref = -Kp*pinv(J)*nu
	q_dot_ref = np.concatenate((np.zeros([6,1]) , qa_dot_ref))
	
	# Computing the updated configuration
	q_ref = pin.integrate(solo.model, q_ref, q_dot_ref * dt)
	qa_ref = q_ref[7:]
	
	solo.display(q)
	
	#################################################
	
	# Parameters for the PD controller
	Kp = 8.
	Kd = 0.2
	torque_sat = 3 # torque saturation in N.m
	torques_ref = np.zeros((8,1)) # feedforward torques

	# Call the PD controller
	torques = PD(qa_ref, qa_dot_ref, qa, qa_dot, dt, Kp, Kd, torque_sat, torques_ref)
   
	# torques must be a numpy array of shape (8, 1) containing the torques applied to the 8 motors
	return torques
