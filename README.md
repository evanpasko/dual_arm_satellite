# Dual Arm Satellite Simulation

A physics simulation and MVP control solution for a free-floating satellite with two 4 DoF robot arms using thruster end-effectors.

## Simplifying Assumptions

In order to achieve the solution in the requested time limit, the following simplifying assumptions were made to the model with brief justifications provided for each: 

### Massless Robot Arms and Effectors

Treating the arms as massless gives a much simpler approach to handling the dynamics of moving joints

### Joints Teleport - no internal torques/position control response

The joints of each robot arm will simply teleport to their requested position (either from teleop or target-pose controller). This removes the work of having to create an internal control/interpolation loop for each joint.

### Instantaneous Thrust

Assuming zero-time linear impulses of the thruster end-effector rather than a more realistic change in momentum over time will help simplify the dynamics a lot when calculating the resulting linear and angular momentum transfer from a burn. 

### Satellite is not in Orbit

Per the problem description (pdf in base directory) the satellite is in 0 gravity so there is no orbital mechanics to include in the physics engine

### No Nearby Bodies

Also assume no nearby small bodies that could induce microgravity


