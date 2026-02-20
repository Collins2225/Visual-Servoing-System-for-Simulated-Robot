# =============================================================================
# PHASE 1 — test_tracker.py
# PURPOSE: Test the ObjectTracker on your webcam (or a video file).
#          Run this BEFORE integrating with the robot simulation.
#          See your detection working in real time with debug overlay.
#
# HOW TO RUN:
#   python test_tracker.py                    ← uses webcam (camera index 0)
#   python test_tracker.py --source video.mp4 ← uses a video file
#
# WHAT TO EXPECT:
#   A window opens showing your camera feed with:
#     RED CROSS    → desired position (image center, where robot aims)
#     GREEN CIRCLE → detected object centroid (current position)
#     YELLOW LINE  → error vector (what IBVS will minimize)
#     TEXT OVERLAY → pixel coordinates and error values
#
# CONTROLS:
#   Press 'q' → quit
#   Press 's' → save current frame as debug_frame.png
# =============================================================================

import cv2
import numpy as np
import sys

# Import our tracker from the vision package
# Make sure you run this from the visual_servoing/ root folder
from vision.object_tracker import ObjectTracker


def main():
    # ------------------------------------------------------------------
    # SETUP — choose input source
    # ------------------------------------------------------------------

    # sys.argv holds command-line arguments.
    # sys.argv[0] = script name, sys.argv[1] = first argument (if given)
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    # source=0 means "use the default webcam (camera index 0)"
    # source="video.mp4" means "read from a video file"

    # cv2.VideoCapture opens the camera or video file.
    # It returns a capture object we'll use to read frames one by one.
    cap = cv2.VideoCapture(source)

    # Check if camera/file opened successfully
    if not cap.isOpened():
        print(f"ERROR: Could not open source: {source}")
        print("  → If using webcam, try changing index: cv2.VideoCapture(1)")
        print("  → If using video file, check the path is correct")
        sys.exit(1)

    # Get frame dimensions from the capture source
    # These are the ACTUAL resolution of our camera feed
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {frame_width}x{frame_height}")
    print("Press 'q' to quit, 's' to save a debug frame")

    # ------------------------------------------------------------------
    # CREATE TRACKER
    # ------------------------------------------------------------------

    # Instantiate our ObjectTracker with color-based detection.
    # method="color" uses HSV masking (what we built in object_tracker.py).
    tracker = ObjectTracker(method="color")

    # ------------------------------------------------------------------
    # MAIN LOOP — read frames and run detection
    # ------------------------------------------------------------------

    while True:
        # cap.read() grabs the next frame from camera/video.
        # Returns: (success_bool, frame_array)
        #   ret   = True if frame was read successfully, False at end of video
        #   frame = numpy array of shape (height, width, 3) in BGR format
        ret, frame = cap.read()

        # If ret is False, we've hit end of video or camera disconnected
        if not ret:
            print("End of video or camera disconnected.")
            break

        # ------------------------------------------------------------------
        # DETECTION — find the object in this frame
        # ------------------------------------------------------------------

        # tracker.detect() returns np.array([cx, cy, area]) or None
        # This is the FEATURE VECTOR 's' — the core of visual servoing.
        features = tracker.detect(frame)

        # Print to console so you can see the raw numbers
        if features is not None:
            cx, cy, area = features
            # Compute error relative to image center
            # This is what the IBVS control law will receive in Phase 3
            ex = cx - (frame_width  // 2)  # >0 means object is RIGHT of center
            ey = cy - (frame_height // 2)  # >0 means object is BELOW center
            print(f"  Detected → cx={cx:.0f}, cy={cy:.0f}, area={area:.0f} | "
                  f"Error → ex={ex:.0f}, ey={ey:.0f}", end='\r')
        else:
            print("  No object detected.                                    ", end='\r')

        # ------------------------------------------------------------------
        # VISUALIZATION — draw debug overlay
        # ------------------------------------------------------------------

        # We draw on a COPY of the frame so the original is untouched
        # (useful if you ever want to process the clean frame separately)
        debug_frame = frame.copy()
        debug_frame = tracker.draw_debug(debug_frame, features)

        # Show the debug frame in a window named "Phase 1 - Object Tracker"
        cv2.imshow("Phase 1 - Object Tracker", debug_frame)

        # ------------------------------------------------------------------
        # KEYBOARD INPUT — cv2.waitKey(1) waits 1ms between frames
        # Returns the ASCII code of the key pressed, or -1 if none
        # ------------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF  # & 0xFF ensures we get a clean byte

        if key == ord('q'):
            # ord('q') = ASCII code 113
            # User wants to quit — break the loop
            print("\nQuitting...")
            break

        elif key == ord('s'):
            # Save the current debug frame as an image file
            filename = "debug_frame.png"
            cv2.imwrite(filename, debug_frame)
            print(f"\nSaved: {filename}")

    # ------------------------------------------------------------------
    # CLEANUP — always release camera and close windows
    # ------------------------------------------------------------------

    # cap.release() tells OpenCV we're done with the camera
    # Without this, the camera light may stay on and other apps can't use it
    cap.release()

    # cv2.destroyAllWindows() closes all OpenCV display windows
    cv2.destroyAllWindows()
    print("Done.")


# This pattern means: only run main() if this file is executed directly,
# NOT if it's imported as a module by another script.
# (Standard Python convention for runnable scripts)
if __name__ == "__main__":
    main()