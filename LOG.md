# Create (Jun. 30 2022) [MotionSimulation_Collection]
This inherits and collects many old projects that generate motion simulation of vehicles and/or humans.

# 20220804
Initial data structure is (V1):
- Environment: 
1. Static:  One image for each scene 
   [dataset - index (of scene/video/etc.) - SE image + Traj CSV]
   Traj CSV: t + id + index + x + y
   The image file name is in the form of “(index).(img)” (or others)
2. Dynamic: One image for each time step
   [dataset - index (of scene/video/etc.) - DE images + Traj CSV]
   Traj CSV: t + id + index + x + y
   The image file name is in the form of “(t).(img)”
- Predict:
1. Position:   Given the prediction time offset, return the future position.
   Dataset CSV: p0~pp + t + id + index + T
   “p_i” is the i-th past position in the form of “x_y_t”. (“t” to find the image)
   “T” is the future position in the form of “x_y_T”. (“T” is the time offset)
2. Trajectory: Given the maximal prediction time offset, return future positions.
   Dataset CSV: p0~pp + t + id + index + T1~Tf
   “p_i” is the i-th past position in the form of “x_y_t”.
   “T_i” is the i-th future position in the form of “x_y”.

# 20220804
Simplified data stucture is (V2):
- Environment: The same
- Predict: Only in the form of trajectories
1. Trajectory:
   Dataset CSV: p0~pp + t + id + index + T1~Tf
   “p_i” is the i-th past position in the form of “x_y_t”.
   “T_i” is the i-th future position in the form of “x_y”.
**NOTE** The position prediction will be a part of trajectory prediction now!
**ADD** “gather_all_data” in "utils_data.py" to substitute old ones.
