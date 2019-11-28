# coding: utf8

#####################
## LOADING MODULES ##
#####################

# PyBullet modules
import pybullet as p # PyBullet simulator
import pybullet_data

# Pinocchio modules
import pinocchio as pin # Pinocchio library
from pinocchio.utils import * # Utilitary functions from Pinocchio
from pinocchio.robot_wrapper import RobotWrapper # Robot Wrapper to load an URDF in Pinocchio

# Other modules
import time # Time module to sleep()
from initialization_simulation import * # Functions to initialize the simulation and retrieve joints positions/velocities
from walking_controller import * # Controller functions

####################
## INITIALIZATION ##
####################

dt = 0.001 # time step of the simulation
realTimeSimulation = True # If True then we will sleep in the main loop to have a 1:1 ratio of (elapsed real time / elapsed time in the simulation)
enableGUI = False # enable PyBullet GUI or not
robotId, solo, revoluteJointIndices = configure_simulation(dt, enableGUI)

###############
## MAIN LOOP ##
###############
	
for i in range(10000): # run the simulation during dt * i_max seconds (simulation time)
   
	# Time at the start of the loop
	if realTimeSimulation:
		t0 = time.clock()

	# Get position and velocity of all joints in PyBullet (free flying base + motors)
	q, qdot = getPosVelJoints(robotId, revoluteJointIndices)

	# Call controller to get torques for all joints
	jointTorques = c_walking_IK(q, qdot, dt, solo, dt*i)[0]

	# Get the configurations q for one cycle T
	Q_list = c_walking_IK(q, qdot, dt, solo, dt*i)[1]
	 
	
	# Set control torques for all joints in PyBullet
	p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

	# Compute one step of simulation
	p.stepSimulation()
	
	# Sleep to get a real time simulation
	if realTimeSimulation:
		t_sleep = dt - (time.clock()-t0)
		if t_sleep > 0:
			time.sleep(t_sleep)

T = 0.4 			# period of the foot trajectory
DT = int(T/dt)		# number of iteration for one period
Q = Q_list[4*DT:5*DT] # Q is the list of the configurations for one trajectory cycle (the 2nd one to avoid the singularities of the 1st one)

print('q[400][8] = ', end='')
print('{', end='') 
for i in range(DT-1): 
	print ('{',"{},{},{},{},{},{},{},{}".format(Q[i][0,0],Q[i][1,0],Q[i][2,0],Q[i][3,0],Q[i][4,0],Q[i][5,0],Q[i][6,0],Q[i][7,0]),'},') 
print ('{',"{},{},{},{},{},{},{},{}".format(Q[399][0,0],Q[399][1,0],Q[399][2,0],Q[399][3,0],Q[399][4,0],Q[399][5,0],Q[399][6,0],Q[399][7,0]),'}', end='')
print('};') 

# Shut down the PyBullet client
p.disconnect()
