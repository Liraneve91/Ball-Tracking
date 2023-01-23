import cv2
import numpy as np
import csv
import time
import argparse


def ball_detector_using_color(frame: np.array) -> (tuple, np.array):
    # Define the lower and upper bounds of the "ping pong" color in the HSV color space
    ping_pong_lower = np.array([0, 150, 200])
    ping_pong_upper = np.array([120, 255, 255])

    # Threshold the frame to only select colors in the "ping pong" range
    mask = cv2.inRange(frame, ping_pong_lower, ping_pong_upper)

    # Use a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Initialize variables for the center of the ball and the radius
    ball_center = None
    ball_radius = None

    # Only proceed if at least one contour was found
    if len(cnts) > 0:
        # Find the largest contour in the mask
        c = max(cnts, key=cv2.contourArea)

        # Use the contour to compute the minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        ball_center = (int(x), int(y))
        ball_radius = int(radius)

        # Draw the circle on the frame
        cv2.circle(frame, ball_center, ball_radius, (0, 255, 0), 2)
    return ball_center, frame


def ball_detector_using_hough(frame: np.array) -> (tuple, np.array):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the frame
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use the HoughCircles function to detect circles in the frame
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=30)
    ball_x = -1
    ball_y = -1
    # Make sure circles were detected
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Iterate over the circles and draw them on the frame
        for (x, y, r) in circles:
            if r < 35 and r > 10:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                ball_x = x
                ball_y = y
    if ball_x != -1:
        return (ball_x, ball_y), frame
    else:
        return None, frame


def add_to_csv(ball_positions, csv_file_name, times):
    # Read the existing data from the CSV file
    with open(csv_file_name, 'r') as csvfile:
        fieldnames = ['Position-x', 'Position-y', 'Velocity-x', 'Velocity-y']
        csvreader = csv.DictReader(csvfile, fieldnames=fieldnames)
        data = list(csvreader)

    if ball_positions[-1] != None:
        if ball_positions[-2] != None:
            data += [{'Position-x': ball_positions[-1][0], 'Position-y': ball_positions[-1][1],
                      'Velocity-x': np.abs(ball_positions[-1][0] - ball_positions[-2][0]) / (times[-1] - times[-2]),
                      'Velocity-y': np.abs(ball_positions[-1][1] - ball_positions[-2][1]) / (times[-1] - times[-2])}]
        else:
            data += [{'Position-x': ball_positions[-1][0], 'Position-y': ball_positions[-1][1], 'Velocity-x': -1,
                      'Velocity-y': -1}]

    else:
        data += [{'Position-x': -1, 'Position-y': -1, 'Velocity-x': -1, 'Velocity-y': -1}]
    # Write the updated data back to the CSV file
    with open(csv_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writerows(data)


def main(args):
    # Initialize the webcam
    cap = cv2.VideoCapture(args.source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.mp4' file.
    video_output_filename = 'Output/output_color.avi'
    out = cv2.VideoWriter(video_output_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                          (frame_width, frame_height))
    csv_file_name = 'Output/data_color.csv'
    txt_file_name = 'Output/statistics_color.txt'
    ret = True
    ball_positions = []
    ball_positions_for_drawing = []
    times = []

    with open(csv_file_name, 'w', newline='') as csvfile:
        # Create a CSV DictWriter object
        fieldnames = ['Position-x', 'Position-y', 'Velocity-x', 'Velocity-y']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write the header row
        csvwriter.writeheader()

    while ret:
        # Read a frame from the webcam
        ret, frame = cap.read()
        times.append(time.time())
        if ret:
            ball_center, frame = ball_detector_using_color(frame)
            # ball_center, frame = ball_detector_using_hough(frame)
            ball_positions.append(ball_center)
            if ball_center != None:
                ball_positions_for_drawing.append(ball_center)

            if len(ball_positions_for_drawing) > 1:
                last_positions = ball_positions_for_drawing[-10:]
                for i, (a, b) in enumerate(zip(last_positions, last_positions[1:])):
                    if a[0] == -1 or b[0] == -1:
                        continue
                    color = (0, 0, 0)
                    thickness = 2
                    cv2.line(frame, a, b, color, thickness)

            add_to_csv(ball_positions, csv_file_name, times)
            # Show the frame to the screen
            cv2.imshow("Frame", frame)
            out.write(frame)
            key = cv2.waitKey(33) & 0xFF

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="Videos/input.mp4", help='source')
    args = parser.parse_args()
    main(args)
