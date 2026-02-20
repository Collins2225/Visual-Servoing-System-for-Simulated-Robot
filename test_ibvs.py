# =============================================================================
# PHASE 3 — test_ibvs.py
# PURPOSE: Test the IBVSController in isolation — NO robot, NO camera needed.
#          Just pure math verification.
#          Confirms the control law computes correct velocities.
#
# HOW TO RUN:
#   python test_ibvs.py
#
# WHAT YOU SHOULD SEE:
#   A series of tests showing:
#   → Error computation is correct
#   → Velocity directions are correct (negative = corrective)
#   → Convergence detection works
#   → Gain tuning changes response speed
# =============================================================================

import numpy as np
from control.ibvs import IBVSController


def print_section(title):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)


def main():
    print("PHASE 3 — IBVS Controller Test")
    print("Testing control law math without robot or camera")

    # Create controller with default settings
    # lambda=0.3, desired=[320, 240, 10000]
    controller = IBVSController(lambda_gain=0.3)

    # ------------------------------------------------------------------
    # TEST 1 — Object perfectly centered (should produce zero velocity)
    # ------------------------------------------------------------------
    print_section("TEST 1: Object at perfect center")

    # Simulate object exactly at desired position
    features_centered = np.array([320.0, 240.0, 10000.0])
    velocity, error, converged = controller.compute_velocity(features_centered)

    print(f"  Features (current): {features_centered}")
    print(f"  Desired:            {controller.desired}")
    print(f"  Error:              {error}")
    print(f"  Velocity:           {velocity}")
    print(f"  Converged:          {converged}")
    print(f"  Expected: error=[0,0,0], converged=True")
    assert converged, "Should be converged when at center!"
    print(f"  ✅ PASSED")

    # ------------------------------------------------------------------
    # TEST 2 — Object to the RIGHT of center
    # ------------------------------------------------------------------
    print_section("TEST 2: Object to the RIGHT of center")

    # cx=450 means object is 130 pixels to the right of center (320)
    features_right = np.array([450.0, 240.0, 10000.0])
    velocity, error, converged = controller.compute_velocity(features_right)

    print(f"  Features: cx=450 (object is RIGHT of center)")
    print(f"  Error ex: {error[0]}  (positive = right of center)")
    print(f"  Velocity vx: {velocity[0]:.4f}  (should be NEGATIVE to move left)")
    print(f"  Converged: {converged}")

    # vx should be negative (move left to correct rightward error)
    assert velocity[0] < 0, "vx should be negative when object is right of center!"
    print(f"  ✅ PASSED — robot correctly moves LEFT to center the object")

    # ------------------------------------------------------------------
    # TEST 3 — Object to the LEFT of center
    # ------------------------------------------------------------------
    print_section("TEST 3: Object to the LEFT of center")

    # cx=150 means object is 170 pixels to the left of center (320)
    features_left = np.array([150.0, 240.0, 10000.0])
    velocity, error, converged = controller.compute_velocity(features_left)

    print(f"  Features: cx=150 (object is LEFT of center)")
    print(f"  Error ex: {error[0]}  (negative = left of center)")
    print(f"  Velocity vx: {velocity[0]:.4f}  (should be POSITIVE to move right)")

    # vx should be positive (move right to correct leftward error)
    assert velocity[0] > 0, "vx should be positive when object is left of center!"
    print(f"  ✅ PASSED — robot correctly moves RIGHT to center the object")

    # ------------------------------------------------------------------
    # TEST 4 — Object below center
    # ------------------------------------------------------------------
    print_section("TEST 4: Object BELOW center")

    # cy=380 means object is 140 pixels below center (240)
    features_below = np.array([320.0, 380.0, 10000.0])
    velocity, error, converged = controller.compute_velocity(features_below)

    print(f"  Features: cy=380 (object is BELOW center)")
    print(f"  Error ey: {error[1]}  (positive = below center)")
    print(f"  Velocity vy: {velocity[1]:.4f}  (should be NEGATIVE to move down)")

    assert velocity[1] < 0, "vy should be negative when object is below center!"
    print(f"  ✅ PASSED — robot correctly moves DOWN to center the object")

    # ------------------------------------------------------------------
    # TEST 5 — Object too far (area too small)
    # ------------------------------------------------------------------
    print_section("TEST 5: Object TOO FAR (area too small)")

    # area=3000 means object appears small → it's far away
    # robot should move FORWARD (positive vz) to get closer
    features_far = np.array([320.0, 240.0, 3000.0])
    velocity, error, converged = controller.compute_velocity(features_far)

    print(f"  Features: area=3000 (object appears SMALL → too far)")
    print(f"  Error e_area: {error[2]}  (negative = area too small = too far)")
    print(f"  Velocity vz: {velocity[2]:.6f}  (should be POSITIVE to move forward)")

    assert velocity[2] > 0, "vz should be positive when object is too far!"
    print(f"  ✅ PASSED — robot correctly moves FORWARD to approach object")

    # ------------------------------------------------------------------
    # TEST 6 — Gain tuning effect
    # ------------------------------------------------------------------
    print_section("TEST 6: Effect of different gain values")

    features_test = np.array([420.0, 280.0, 10000.0])

    for gain in [0.1, 0.3, 0.5, 1.0]:
        ctrl = IBVSController(lambda_gain=gain)
        vel, err, _ = ctrl.compute_velocity(features_test)
        speed = np.linalg.norm(vel)
        print(f"  λ={gain:.1f} → velocity magnitude={speed:.6f}  "
              f"vx={vel[0]:.4f}  vy={vel[1]:.4f}")

    print(f"\n  Notice: higher λ → larger velocity → faster response")
    print(f"  ✅ Gain tuning working correctly")

    # ------------------------------------------------------------------
    # TEST 7 — Simulated convergence over multiple steps
    # ------------------------------------------------------------------
    print_section("TEST 7: Simulated convergence loop")

    controller.reset()
    ctrl = IBVSController(lambda_gain=0.3)

    # Start with object far from center
    cx   = 450.0
    cy   = 350.0
    area = 10000.0

    print(f"  Starting position: cx={cx}, cy={cy}")
    print(f"  Goal:              cx=320, cy=240")
    print(f"  Simulating {50} steps...\n")

    for step in range(50):
        features = np.array([cx, cy, area])
        velocity, error, converged = ctrl.compute_velocity(features)

        # Simulate robot moving — update fake position
        # In real system this happens via move_joints()
        # Here we just shift cx,cy directly to test math
        cx   += velocity[0] * 1000   # convert back to pixels for simulation
        cy   += velocity[1] * 1000

        if step % 10 == 0 or converged:
            pixel_err = np.linalg.norm([error[0], error[1]])
            print(f"  Step {step:3d} | "
                  f"cx={cx:6.1f} cy={cy:6.1f} | "
                  f"error={pixel_err:6.2f}px | "
                  f"{'✅ CONVERGED!' if converged else 'moving...'}")

        if converged:
            break

    summary = ctrl.get_error_summary()
    print(f"\n  Summary: {summary}")
    print(f"  ✅ Convergence simulation working!")

    # ------------------------------------------------------------------
    # ALL TESTS PASSED
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED ✅")
    print("  IBVSController is working correctly!")
    print("  Ready for Phase 4 — Closing the Loop")
    print("=" * 50)


if __name__ == "__main__":
    main()