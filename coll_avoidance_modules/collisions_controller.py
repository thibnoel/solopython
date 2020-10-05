import numpy as np

# Compute a viscoelastic repulsive torque for a list of collisions results (distances + jacobians)
def computeRepulsiveTorque(q, vq, collDistances, collJacobians, dist_thresh=0.1, kp=0, kv=0, active_pairs=[]):
    # Initialize repulsive torque
    tau_avoid = np.zeros(len(q))

    # Initialize active pairs if unspecified
    if(len(active_pairs) == 0):
        active_pairs = [i for i in range(len(collDistances))]

    # Loop through the distance to check for threshold violation
    for i in range(len(collDistances)):
        J = collJacobians[i]
        d = collDistances[i]

        tau_rep = np.zeros(len(q))
        # If violation, compute viscoelastic repulsive torque along the collision jacobian
        if(d < dist_thresh and i in active_pairs):
            tau_rep = -kp*(d - dist_thresh) - kv*J@vq
        tau_avoid += tau_rep*J.T
    
    return tau_avoid


def computeEmergencyTorque(vq, kv):
    return -kv*vq


# Compute a condition to switch to the emergency behavior
# How to deal with v ? 
def emergencyCondition(collDistances, vq, tau_q, d_thresh, tau_thresh):
    if(np.min(collDistances) < d_thresh or np.max(np.abs(tau_q)) > tau_thresh):
        return True
    return False