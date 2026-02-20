# =============================================================================
# PHASE 1 — vision/object_tracker.py
# PURPOSE: Detect an object in a camera frame and return its position
#          as a feature vector [cx, cy, area] in IMAGE SPACE (pixels).
#
# KEY CONCEPT — Image Space vs World Space:
#   World Space → real 3D coords in meters:  (x=0.5m, y=0.1m, z=0.3m)
#   Image Space → 2D pixel coords:           (cx=320px, cy=240px)
#
#   Visual Servoing works ENTIRELY in image space.
#   We never say "object is 0.5m away."
#   We say "object centroid is at pixel (320, 240)."
#   The robot moves until the object reaches the DESIRED pixel (usually center).
# =============================================================================

import cv2       # OpenCV — camera, image processing, contours, drawing
import numpy as np  # NumPy — array math on pixel data


class ObjectTracker:
    """
    Detects a colored object in a camera frame.
    Returns a 3-element feature vector: [cx, cy, area]
      cx   → horizontal center of object in pixels  (left=0, right=640)
      cy   → vertical center of object in pixels    (top=0, bottom=480)
      area → pixel area of object blob              (larger = closer to camera)
    """

    def __init__(self, method="color"):
        """
        method: "color" uses HSV masking (what we build now)
                "aruco" uses ArUco markers (Phase 3 upgrade)
        """
        self.method = method  # store method for use in detect()

        # --- HSV COLOR RANGE for a YELLOW target object ---
        # WHY HSV instead of BGR?
        #   BGR mixes color + brightness together
        #   → lighting changes wreck detection
        #   HSV separates Hue (color) from Saturation+Value (brightness)
        #   → robust to lighting changes
        #
        # HSV channels:
        #   H (Hue)        0–179  → actual color    yellow ≈ 20–35
        #   S (Saturation) 0–255  → vividness        100+ means not washed out
        #   V (Value)      0–255  → brightness        100+ means not too dark
        #
        # So [20, 100, 100] → [35, 255, 255] means:
        #   "anything yellow-ish, vivid, and visible"
        self.lower_hsv = np.array([0, 0, 0])  
        self.upper_hsv = np.array([179, 50, 50])  

    # -------------------------------------------------------------------------
    def detect(self, frame):
        """
        Main detection method. Called every frame in the control loop.

        Args:
            frame: BGR image from camera (numpy array, shape HxWx3)

        Returns:
            np.array([cx, cy, area]) if object found
            None                     if object not visible
        """

        # STEP 1 — Convert BGR → HSV
        # OpenCV reads/stores images as BGR (Blue-Green-Red), NOT RGB.
        # This is a famous OpenCV gotcha — always remember this.
        # We convert to HSV so our color range thresholding is lighting-robust.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # STEP 2 — Create a binary mask
        # inRange() checks every pixel: is it within [lower_hsv, upper_hsv]?
        # Output is a BINARY IMAGE (same size as frame) where:
        #   white (255) = pixel IS our target color  → "this is our object"
        #   black (0)   = pixel is NOT target color  → "this is background"
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # STEP 3 — Morphological Opening (noise removal)
        # Real camera masks have noise: tiny white blobs from reflections, etc.
        # Morphological OPEN = EROSION then DILATION
        #
        #   Erosion  → shrinks white regions → kills small noise specks
        #   Dilation → grows white regions back → restores real object size
        #   Net effect: small noise disappears, large real objects survive
        #
        # np.ones((5,5)) is the KERNEL — defines neighborhood size.
        # 5×5 = aggressive enough to kill noise, gentle enough for real objects.
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # STEP 4 — Find contours (outlines of white blobs)
        # findContours traces the border of every white region in the mask.
        # Returns a LIST of contours. Each contour = array of (x,y) boundary points.
        #
        # cv2.RETR_EXTERNAL  → only outer boundaries, ignore holes inside blobs
        # cv2.CHAIN_APPROX_SIMPLE → compress straight edges to endpoints only
        #                           (saves memory vs storing every point)
        #
        # The '_' discards hierarchy info — we don't need it.
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # STEP 5 — Early exit if nothing detected
        # If the list of contours is empty, no blobs were found → object not visible.
        # The control loop MUST handle None gracefully (robot holds position).
        if not contours:
            return None

        # STEP 6 — Pick the largest contour
        # Multiple blobs may exist (reflections, other yellow objects).
        # We assume the LARGEST blob is our target (most reliable heuristic).
        # max() with key=cv2.contourArea returns the contour with biggest area.
        largest_contour = max(contours, key=cv2.contourArea)

        # STEP 7 — Minimum area threshold
        # Even the biggest blob might be tiny noise if object is out of frame.
        # If area < 300 pixels², it's too small to trust → return None.
        # 300 px² ≈ a ~17×17 pixel square — tune this for your object/camera.
        if cv2.contourArea(largest_contour) < 300:
            return None

        # STEP 8 — Compute centroid using Image Moments
        # A "moment" is a weighted sum of pixel positions.
        # cv2.moments() returns a dictionary of moment values.
        #
        # Key moments:
        #   M["m00"] = total area (sum of all pixels in contour)
        #   M["m10"] = sum of (x * pixel_value)  ← x-weighted area
        #   M["m01"] = sum of (y * pixel_value)  ← y-weighted area
        #
        # Centroid formula (center of mass):
        #   cx = m10 / m00   ← average x position across all pixels
        #   cy = m01 / m00   ← average y position across all pixels
        #
        # This gives the EXACT geometric center of our blob.
        M = cv2.moments(largest_contour)

        # Guard against division by zero (degenerate contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])  # centroid x in pixels
        cy = int(M["m01"] / M["m00"])  # centroid y in pixels

        # STEP 9 — Compute area
        # Area in pixels² = proxy for DEPTH (distance from camera).
        # Larger blob → object is closer to camera.
        # Smaller blob → object is farther away.
        # This lets us control the robot's Z-axis (approach/retreat)
        # WITHOUT needing a depth sensor!
        area = cv2.contourArea(largest_contour)

        # STEP 10 — Return the feature vector
        # [cx, cy, area] is our VISUAL FEATURE VECTOR 's' in IBVS notation.
        # The control law will minimize: error = s - s*
        # where s* = [desired_cx, desired_cy, desired_area]
        # (usually image center + a target area for desired grasping distance)
        return np.array([cx, cy, area], dtype=float)

    # -------------------------------------------------------------------------
    def draw_debug(self, frame, features):
        """
        Draw detection overlay on frame for debugging.
        Call this AFTER detect() to visualize what the tracker sees.

        Args:
            frame:    BGR image (will be drawn on — pass a copy if needed)
            features: result from detect() — np.array([cx, cy, area]) or None

        Returns:
            frame with overlay drawn on it
        """
        h, w = frame.shape[:2]

        # Draw the DESIRED position (image center) as a red cross
        # This is where we WANT the object to be (s* in IBVS)
        desired_cx, desired_cy = w // 2, h // 2
        cv2.drawMarker(
            frame,
            (desired_cx, desired_cy),
            color=(0, 0, 255),        # red in BGR
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )
        cv2.putText(
            frame, "DESIRED", (desired_cx + 10, desired_cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

        if features is not None:
            cx, cy, area = int(features[0]), int(features[1]), features[2]

            # Draw detected centroid as a green filled circle
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)  # -1 = filled

            # Draw error line from detected → desired (this is what IBVS minimizes)
            cv2.line(frame, (cx, cy), (desired_cx, desired_cy), (255, 255, 0), 2)

            # Compute pixel error (will be passed to control law later)
            ex = cx - desired_cx  # positive = object is to the RIGHT of center
            ey = cy - desired_cy  # positive = object is BELOW center

            # Display stats on frame
            cv2.putText(
                frame, f"Detected: ({cx}, {cy})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Error: ex={ex}, ey={ey}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            cv2.putText(
                frame, f"Area (depth proxy): {area:.0f} px2",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2
            )
        else:
            # No detection — warn the user
            cv2.putText(
                frame, "NO TARGET DETECTED",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        return frame