import cv2
import numpy as np
import time
import random

from vision.object_tracker   import ObjectTracker
from simulation.pybullet_env import RobotEnv
from control.ibvs            import IBVSController


def main():
    print("Visual Servoing System - Phase 4")
    print("Closing the Loop")
    print("=" * 50)

    # ── Initialize all 3 components ──────────────────────────────────
    # Phase 2: virtual robot world
    env = RobotEnv(gui=True)

    # Phase 1: object detector
    tracker = ObjectTracker(method="color")

    # Tune tracker for PyBullet sphere color (grey/white)
    tracker.lower_hsv = np.array([0,   0,   150])
    tracker.upper_hsv = np.array([179, 60,  255])

    # Phase 3: IBVS control law
    controller = IBVSController(lambda_gain=0.3)

    # ── Move robot to home position ───────────────────────────────────
    print("Moving to home position...")
    home_pos    = [0.4, 0.0, 0.6]
    home_joints = env.compute_ik(home_pos)

    for _ in range(200):
        env.move_joints(home_joints)
        time.sleep(1/240)

    print("Home position reached!")
    print("Visual servoing started!")
    print("Controls: q=quit  r=random target  +=faster  -=slower")
    print("-" * 50)

    frame_count = 0
    gain        = 0.3

    while True:
        # STEP 1 - render camera frame from robot hand
        # returns BGR image same format as webcam
        frame = env.get_camera_image()

        # STEP 2 - detect object in frame
        # returns [cx, cy, area] or None
        features = tracker.detect(frame)

        # STEP 3 - compute velocity and move robot
        velocity  = np.zeros(3)
        error     = np.zeros(3)
        converged = False

        if features is not None:

            # control law: velocity = -lambda x error
            velocity, error, converged = controller.compute_velocity(features)

            if converged:
                # target reached - hold current position
                ee_pos, _ = env.get_ee_pose()
                env.move_joints(env.compute_ik(list(ee_pos)))
                status_msg = "TARGET REACHED!"

            else:
                # STEP 4 - apply velocity to end effector position
                ee_pos, ee_orn = env.get_ee_pose()
                ee_pos = list(ee_pos)

                # move hand by velocity amount each frame
                # each step is a small movement in the right direction
                ee_pos[0] += velocity[0]
                ee_pos[1] += velocity[1]
                ee_pos[2] += velocity[2]

                # clamp to safe workspace limits
                # prevents robot reaching impossible positions
                ee_pos[0] = np.clip(ee_pos[0],  0.2,  0.8)
                ee_pos[1] = np.clip(ee_pos[1], -0.4,  0.4)
                ee_pos[2] = np.clip(ee_pos[2],  0.2,  0.9)

                # solve IK to get joint angles then move
                new_joints = env.compute_ik(ee_pos)
                env.move_joints(new_joints)
                status_msg = "SERVOING..."

        else:
            # no detection - keep physics running, hold position
            env.step()
            status_msg = "NO TARGET"

        # STEP 5 - draw debug overlay on frame
        debug_frame = frame.copy()
        debug_frame = tracker.draw_debug(debug_frame, features)

        if features is not None:
            ex        = int(error[0])
            ey        = int(error[1])
            pixel_err = np.linalg.norm([ex, ey])

            # show error values
            cv2.putText(debug_frame,
                f"Error: ex={ex}px  ey={ey}px  mag={pixel_err:.1f}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # show velocity being sent to robot
            cv2.putText(debug_frame,
                f"Velocity: vx={velocity[0]:.3f}  vy={velocity[1]:.3f}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

        # show status message
        # green = converged, yellow = servoing, red = no target
        color = (0, 255, 0) if converged else (255, 255, 0) if features is not None else (0, 0, 255)
        cv2.putText(debug_frame, status_msg,
            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # show gain and frame count
        cv2.putText(debug_frame,
            f"Gain: {gain:.1f}  Frame: {frame_count}",
            (10, debug_frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Visual Servoing - Robot Eye View", debug_frame)

        # print to terminal every 30 frames
        if frame_count % 30 == 0:
            if features is not None:
                px_err = np.linalg.norm([error[0], error[1]])
                print(f"Frame {frame_count:4d} | Error: {px_err:6.1f}px | {status_msg}")
            else:
                print(f"Frame {frame_count:4d} | No detection")

        # STEP 6 - keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting...")
            break

        elif key == ord('r'):
            # move target to random position - watch robot re-servo!
            new_target = [
                random.uniform(0.3, 0.7),
                random.uniform(-0.3, 0.3),
                random.uniform(0.3, 0.7)
            ]
            env.reset_target(new_target)
            controller.reset()
            print(f"Target moved to: {[round(x, 2) for x in new_target]}")

        elif key == ord('+') or key == ord('='):
            # increase gain - robot reacts faster
            gain = min(gain + 0.1, 1.0)
            controller.set_gain(gain)
            print(f"Gain increased to: {gain:.1f}")

        elif key == ord('-'):
            # decrease gain - robot reacts slower but more stable
            gain = max(gain - 0.1, 0.1)
            controller.set_gain(gain)
            print(f"Gain decreased to: {gain:.1f}")

        frame_count += 1

    # ── Cleanup ───────────────────────────────────────────────────────
    summary = controller.get_error_summary()
    print("\nSession summary:")
    if isinstance(summary, dict):
        for k, v in summary.items():
            print(f"  {k}: {v}")

    cv2.destroyAllWindows()
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()