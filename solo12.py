import numpy as np
from utils.abstractRobotHal import RobotHAL

class Solo12(RobotHAL):
    ''' Define the hardware interface to solo12'''

    def __init__(self, interfaceName="", dt=0.001):
        RobotHAL.__init__(self, interfaceName, dt)

    def InitRobotSpecificParameters(self):
        ''' Definition of the Solo12 paramters '''
        self.nb_motors = 12
        self.motorToUrdf = [0, 3, 2, 1, 5, 4, 6, 9, 8, 7, 11, 10]
        self.gearRatio = np.array(self.nb_motors * [9., ])  # gearbox ratio
        self.motorKt = np.array(self.nb_motors * [0.025, ])  # Nm/A
        self.motorSign = np.array([-1, +1, -1, -1, +1, +1,
                                   -1, +1, -1, -1, +1, +1])
        self.maximumCurrent = 5.0  # A
        # To get this offsets, run the calibration with self.encoderOffsets at 0,
        # then manualy move the robot in zero config, and paste the position here (note the negative sign!)
        #self.encoderOffsets = - np.array([1.940310, -2.658198, -2.893262, -1.918342,
        #                                  1.501880, -0.694378, 2.974716, -0.415520, -0.326077, 0.444250, 2.433491, -0.384136]) # Solo12 of Max Planck
        # self.encoderOffsets = - np.array([-1.7298685312271118, 1.5541203022003174, -0.22502104938030243, -2.7335188388824463,
        #                                  2.563904047012329, -2.5461583137512207, 1.5116088390350342, -2.1389687061309814, -1.7647048234939575, 0.12051336467266083, -3.485957145690918, -0.834662675857544])
        #self.encoderOffsets = - np.array([-1.7356888055801392, 1.8896119594573975, -0.31902867555618286, -2.6425065994262695,
        #                                  2.7726054191589355, -2.7252161502838135, 1.5070443153381348, -2.3966293334960938, 0.2065046727657318, -1.5028218030929565, 3.048154592514038, -1.1061315536499023])
        self.encoderOffsets = - np.array([-1.743848204612732, 1.8073862791061401, -0.38901469111442566, -2.7215936183929443, 2.8377127647399902, -2.721593141555786, 1.2487566471099854,
                                          -2.0542333126068115, 1.7747472524642944, -1.5836209058761597, 2.703704833984375, -0.7676450610160828])
        
        # self.encoderOffsets *= 0.
        # 180 degree roll
        # self.rotateImuVectors = lambda x: [-x[0], -x[1], x[2]]
        # self.rotateImuOrientation = lambda q: [q[3], -q[2], -q[1], q[0]]

        # IMU oriented like the base (no rotation)
        self.rotateImuVectors = lambda x: [x[0], x[1], x[2]]
        self.rotateImuOrientation = lambda q: [q[3], -q[2], q[1], -q[0]]