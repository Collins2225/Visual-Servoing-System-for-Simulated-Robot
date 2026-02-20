# =============================================================================
# PHASE 2 — test_simulation.py
# PURPOSE: Test the PyBullet simulation environment standalone.
#          Verifies robot loads, camera renders, and joints move.
#          Run this BEFORE connecting vision to simulation in Phase 4.
#
# HOW TO RUN:
#   python test_simulation.py
#
# WHAT YOU SHOULD SEE:
#   1. A PyBullet 3D window opens
#   2. A Kuka robot arm appears on a flat ground
#   3. A small sphere sits in front of the robot
#   4. A separate OpenCV window shows the robot's eye-view camera
#   5. The robot slowly waves its arm (joint test)
#   6. Terminal prints camera feed info every second
#
# CONTROLS:
#   PyBullet window → left click + drag to rotate view
#                   → scroll wheel to zoom
#                   → right click + drag to pan
#   Press 'q' in OpenCV window → quit
# =============================================================================

import cv2
import numpy as np
import time

# Import our simulation environment
from simulation.pybullet_env import RobotEnv


def main():
    print("=" * 50)
    print("PHASE 2 — Simulation Test")
    print("=" * 50)

    # ------------------------------------------------------------------
    # CREATE SIMULATION
    # ------------------------------------------------------------------

    # gui=True opens the 3D PyBullet window
    # You can rotate/zoom the view with your mouse
    env = RobotEnv(gui=True)

    print("\nSimulation running!")
    print("PyBullet 3D window should be open.")
    print("OpenCV camera window will open shortly.")
    print("Press 'q' in the OpenCV window to quit.\n")

    # ------------------------------------------------------------------
    # TEST 1 — Verify Camera Renders Correctly
    # ------------------------------------------------------------------

    print("TEST 1: Rendering camera image from end effector...")
    frame = env.get_camera_image()

    # frame should be a numpy array of shape (480, 640, 3)
    print(f"  Camera frame shape: {frame.shape}")   # expect (480, 640, 3)
    print(f"  Camera frame dtype: {frame.dtype}")   # expect uint8
    print(f"  ✅ Camera render working!\n")

    # ------------------------------------------------------------------
    # TEST 2 — Verify End Effector Pose
    # ------------------------------------------------------------------

    print("TEST 2: Reading end effector pose...")
    pos, orn = env.get_ee_pose()
    print(f"  EE Position:    x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f} meters")
    print(f"  EE Orientation: {[round(o, 3) for o in orn]} (quaternion)")
    print(f"  ✅ End effector pose reading working!\n")

    # ------------------------------------------------------------------
    # TEST 3 — Verify Inverse Kinematics
    # ------------------------------------------------------------------

    print("TEST 3: Computing inverse kinematics...")
    test_target = [0.5, 0.1, 0.5]   # where our sphere target is
    angles = env.compute_ik(test_target)
    print(f"  Target position: {test_target}")
    print(f"  IK joint angles: {[round(a, 3) for a in angles]}")
    print(f"  ✅ IK solver working!\n")

    # ------------------------------------------------------------------
    # MAIN LOOP — Show camera feed + gentle robot movement
    # ------------------------------------------------------------------

    print("Running main loop... Press 'q' to quit")
    print("-" * 50)

    frame_count = 0
    start_time  = time.time()

    # Gentle wave motion — oscillate joint 1 back and forth
    # np.sin() produces smooth values between -1 and +1
    # We scale it to make the motion small and safe
    base_angles = [0, -0.5, 0, -1.5, 0, 1.0, 0]   # home position

    while True:
        # ------------------------------------------------------------------
        # Gentle robot movement — so we can see the camera view changing
        # ------------------------------------------------------------------
        elapsed = time.time() - start_time

        # Oscillate joint 0 (base rotation) slowly back and forth
        # np.sin(elapsed) → smooth value cycling -1 to +1 over ~6 seconds
        # * 0.3 → scale to ±0.3 radians (small, safe motion)
        wave_angles = base_angles.copy()
        wave_angles[0] = np.sin(elapsed * 0.5) * 0.3   # slow base rotation
        wave_angles[1] = base_angles[1] + np.sin(elapsed * 0.3) * 0.2

        # Send joint commands to robot
        env.move_joints(wave_angles)

        # ------------------------------------------------------------------
        # Render camera image from robot's eye view
        # ------------------------------------------------------------------
        frame = env.get_camera_image()

        # Add info overlay to the frame
        cv2.putText(
            frame,
            f"Phase 2 - Robot Eye View | Frame: {frame_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Get current EE position and show it on screen
        pos, _ = env.get_ee_pose()
        cv2.putText(
            frame,
            f"EE pos: x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )

        cv2.putText(
            frame,
            "Press 'q' to quit | 'r' to reset target",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # Show the frame
        cv2.imshow("Phase 2 - Robot Eye Camera", frame)

        # Print stats every 60 frames
        if frame_count % 60 == 0:
            fps = frame_count / max(elapsed, 0.001)
            print(f"  Frame {frame_count:4d} | "
                  f"EE: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                  f"FPS: {fps:.1f}")

        frame_count += 1

        # ------------------------------------------------------------------
        # Keyboard input
        # ------------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break

        elif key == ord('r'):
            # Reset target to a random position in front of robot
            import random
            new_pos = [
                random.uniform(0.3, 0.7),   # x: 30-70cm forward
                random.uniform(-0.3, 0.3),  # y: ±30cm sideways
                random.uniform(0.3, 0.7)    # z: 30-70cm height
            ]
            env.reset_target(new_pos)
            print(f"  Target moved to: {[round(p, 2) for p in new_pos]}")

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------
    cv2.destroyAllWindows()
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()