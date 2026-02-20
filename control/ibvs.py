import numpy as np


class IBVSController:
    def __init__(self, lambda_gain=0.3, img_width=640, img_height=480):
        # lambda = how aggressively robot reacts to error
        # 0.1 = slow, 0.3 = balanced, 1.0 = fast but may overshoot
        self.lam = lambda_gain

        # Desired feature vector s* = where we WANT the object
        # center of image = (320, 240), area=10000 = desired depth
        self.desired = np.array([
            img_width  / 2,
            img_height / 2,
            10000.0
        ])

        # Stop moving when pixel error is below this value
        self.threshold = 5.0

        # Store error over time for debugging
        self.error_history = []

    def compute_error(self, features):
        # error = s - s*
        # ex    = cx   - 320    (positive = object right of center)
        # ey    = cy   - 240    (positive = object below center)
        # earea = area - 10000  (negative = too far, positive = too close)
        return features - self.desired

    def compute_velocity(self, features):
        # Step 1 - compute error
        error  = self.compute_error(features)
        ex     = error[0]
        ey     = error[1]
        e_area = error[2]

        # Step 2 - apply control law: velocity = -lambda x error
        # NEGATIVE because we want to move AGAINST the error
        # Scale factors convert pixels to meters
        vx = -self.lam * ex     * 0.001
        vy = -self.lam * ey     * 0.001
        vz = -self.lam * e_area * 0.000005

        velocity = np.array([vx, vy, vz])

        # Step 3 - check convergence
        # norm = sqrt(ex^2 + ey^2) = straight line pixel distance to center
        pixel_error = np.linalg.norm([ex, ey])
        converged   = pixel_error < self.threshold

        # Step 4 - log error
        self.error_history.append(pixel_error)

        return velocity, error, converged

    def is_converged(self, features):
        error       = self.compute_error(features)
        pixel_error = np.linalg.norm([error[0], error[1]])
        return pixel_error < self.threshold

    def set_desired(self, cx, cy, area=10000.0):
        self.desired = np.array([cx, cy, area])

    def set_gain(self, lambda_gain):
        self.lam = lambda_gain

    def reset(self):
        self.error_history = []

    def get_error_summary(self):
        if not self.error_history:
            return "No data yet."
        return {
            "total_frames" : len(self.error_history),
            "initial_error": round(self.error_history[0],  2),
            "final_error"  : round(self.error_history[-1], 2),
            "min_error"    : round(min(self.error_history), 2),
            "max_error"    : round(max(self.error_history), 2),
            "converged"    : self.error_history[-1] < self.threshold
        }
