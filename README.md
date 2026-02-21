This project implements a vision-based closed-loop control system for a simulated robotic arm. 
The robot uses camera feedback to detect and track a target object and adjusts its motion in real time to minimize positional error.

The system demonstrates the integration of computer vision, control systems, and robotics simulation in a fully software-based environment.

Objective

To design and implement an Image-Based Visual Servoing (IBVS) pipeline where:

1.A virtual camera captures the environment.

2.The target object is detected using computer vision.

3.Image error is computed relative to a desired position.

4.A control law generates motion commands.

5.The robot updates its joint positions to reduce error.

Technologies Used

 -Python

 -OpenCV

 -PyBullet (robot simulation)

 -NumPy

Key Features

Real-time object tracking

 -Closed-loop visual feedback control

 -Image error computation

 -Proportional/PID-based control implementation

 -Fully simulated robotic manipulation

Outcome

This project demonstrates how perception and control can be integrated to create intelligent robotic systems 
capable of responding dynamically to visual input — a fundamental concept in modern AI-driven robotics.
“This is a robot that can guide itself using vision. It watches an object through a camera, calculates how far off it is, and moves its arm to correct the error. The whole system runs in simulation, so I can test intelligent robot behavior safely.”
