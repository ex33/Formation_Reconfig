# Formation_Reconfig
Part of an initial Project in trying to investigate Non-Convex optimization problems with Analytical Solutions under the Space Imaging and Optical Systems (SIOS) lab by Professor Savransky. Unfortunately, this project ended up producing a trivial result such that it returned back to the basic root-finding method for optimized injection points between two relative motion orbits. 

These files are functions that build on each other in order to provide a way to validate results and were the product of work done over half a semester.

Hill_Eq:

General Clohessy Wiltshire Equation propagator. Given an initial state (x0 v0) and time span of 0:tf+t_step:t_step, will solve for state at each point. Can then plot results. This is the backbone of all following functions that utilize the closed-form solution of the hill equations.

No_Drift_Centered_Conditions: 

This starts to get into the geometric constraints of the optimization problem. Given the relative motion of the two spacecraft, the relative motion sees a drift in the radial track direction (Y), almost like a spiral through 3-d space. Additionally, looking at the radial and intract direction (X-Y), an ellipse is formed, with its center often times not being the point (0,0). Using the equations found in Vallado, David A. - Fundamentals of Astrodynamics and Applications, one can set constraints on the initial states such that it will return a "stable" relative motion such that it will not drift and has its ellipses centered. In 3d space, having these two constraints satisfied will result in ellipses rather than a spiral. 

Delta_V_Calculator:

This employs the state transformation matrix form of the Hill equations in order to calculate the delta-v necessary to go from one relative motion trajectory to another. Given the initial states and desired final position, one can use an order of matrix inverses and multiplications involving the state transformation matrices to calculate the desired initial velocity to get to the final position, in which the delta-v can be found for this first maneuver using the initial state velocity. To go one step further, the final velocity at the final position can also be calculated using the matrices.

Impulsive_Maneuver:

Given the initial state and desired final state, calculate the total delta-v necessary to go from one to the other. In this use case, both states satisfy the No drift and centered conditions, which makes it easy to see why this would be useful for station-keeping applications as well as Rendevous. This function calls upon Delta_V_Calculator as well as Hill_Eq to first calculate the initial and final states of the transfer trajectory, and then propagate it for plotting purposes. 

Maneuver_Cost(Iterative):

This is starting to use all the previous functions to optimize fuel costs for making this sort of transfer via discretizing an orbit by its phase and minimizing the delta-v via sci.py's root-finding function
