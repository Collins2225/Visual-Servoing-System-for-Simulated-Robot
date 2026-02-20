import cv2
import numpy as np


class ObjectTracker:
    def __init__(self, method="color"):
        self.method = method
        # Change these values based on your object color
        # Current: tuned for BLACK objects
        self.lower_hsv = np.array([0,   0,   0])
        self.upper_hsv = np.array([179, 50,  50])

    def detect(self, frame):
        # Step 1 - Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 2 - Create binary mask
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Step 3 - Remove noise
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Step 4 - Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 5 - Return None if nothing found
        if not contours:
            return None

        # Step 6 - Pick largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Step 7 - Ignore if too small
        if cv2.contourArea(largest_contour) < 300:
            return None

        # Step 8 - Compute centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Step 9 - Compute area (depth proxy)
        area = cv2.contourArea(largest_contour)

        # Step 10 - Return feature vector
        return np.array([cx, cy, area], dtype=float)

    def draw_debug(self, frame, features):
        h, w = frame.shape[:2]
        desired_cx, desired_cy = w // 2, h // 2

        # Draw desired position (red cross)
        cv2.drawMarker(
            frame,
            (desired_cx, desired_cy),
            color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )

        if features is not None:
            cx  = int(features[0])
            cy  = int(features[1])
            area = features[2]

            # Draw detected centroid (green circle)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Draw error line
            cv2.line(frame, (cx, cy), (desired_cx, desired_cy), (255, 255, 0), 2)

            # Compute error
            ex = cx - desired_cx
            ey = cy - desired_cy

            # Display info
            cv2.putText(frame, f"Detected: ({cx}, {cy})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Error: ex={ex}, ey={ey}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Area: {area:.0f} px2",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        else:
            cv2.putText(frame, "NO TARGET DETECTED",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame