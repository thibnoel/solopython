import numpy as np

# Compute a viscoelastic repulsive torque for a list of collisions results (distances + jacobians)
def computeRepulsiveTorque(q, vq, collDistances, collJacobians, dist_thresh=0.1, kp=0, kv=0, active_pairs=[], opposeJacIfNegDist=False):
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
        if(d<0 and opposeJacIfNegDist):
            J = -J

        if(d < dist_thresh and i in active_pairs):
            tau_rep = -kp*(d - dist_thresh) - kv*J@vq
        else:
            tau_rep = 0
        tau_avoid += tau_rep*J.T
    
    return tau_avoid



# Compute a condition to switch to the emergency behavior
# How to deal with v ? 

def emergencyCondition(q, vq, tau_q, q_bounds, vq_max, tau_max):
    for i in range(len(q)):
        if(q[i] < q_bounds[0] or q[i] > q_bounds[1]):
            return 1
    
    for i in range(len(vq)):
        if(np.abs(vq[i]) > vq_max):
            return 2
    
    for i in range(len(tau_q)):
        if(np.abs(tau_q[i]) > tau_max):
            return 3

    return 0