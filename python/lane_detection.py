"""
Copyright 2019 Shivang Patel

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""
import cv2
import numpy


def image_preprocessing(frame):
    """
    Performs preprocessing operation on the image.

    Preprocessing operation such as color -> grayscale, gaussian
    blur and canny edge detection.

    Parameters
    ----------
    frame : numpy.array
        Image frame
    
    Returns
    -------
    numpy.array
        Processed images
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.Canny(numpy.uint8(image), 100, 255)
    return image


def extract_roi(frame, polygon_points_array):
    """
    Extracts the Region of interest of frame using the polygon of points in polygon_points_array

    Parameters
    ----------
    frame : numpy.array
        Image frame
    polygon_points_array : numpy.array
        Contains series of points of a polygon
    """
    black_mask = numpy.zeros_like(frame)
    cv2.fillPoly(black_mask, [polygon_points_array], 255)
    out_image = numpy.zeros_like(frame)
    out_image[black_mask == 255] = frame[black_mask == 255]
    return out_image


def split_hough_lines(lines):
    """
    This method split left lane line and right lane line. Returns slope and intercept of the line

    Parameters
    ----------
    lines : numpy.array
        Contains numpy array of x and y coordinates of line segments
    
    Returns
    -------
    list
        list of slope and intercept of the left lane
    list
        list of slope and intercept of the right lane
    """
    left_lane_line = []
    right_lane_line = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            params = numpy.polyfit((x1, x2), (y1, y2), 1)
            slope = params[0]
            intercept = params[1]
            if slope < 0:
                left_lane_line.append([slope, intercept])
            else:
                right_lane_line.append([slope, intercept])
            pass
        pass
    return left_lane_line, right_lane_line


def construct_common_line(avg_line_params, image_shape):
    """
    This method constructs common line from the average of slope and intercept of the respective lanes

    Parameters
    ----------
    avg_line_params : list
        list containing average of slope and intercept
    image_shape : tuple
        tuple of shape of the image
    
    Returns
    -------
    list:
        List constains x and y coordinates of common line
    """
    avg_slope = avg_line_params[0]
    avg_intercept = avg_line_params[1]
    y1 = image_shape[0]
    y2 = int(y1*(7/10))
    x1 = int((y1 - avg_intercept)/avg_slope)
    x2 = int((y2 - avg_intercept)/avg_slope)
    return [x1, y1, x2, y2]


def draw_hough_lines(original_image, frame):
    """
    This method calculats lines, construct single line out of many lines and
    draws the line to the image.

    Parameters
    ----------
    original_image : numpy.array
        Contains original image
    frame : numpy.array
        Contains processed image
    """
    lines = cv2.HoughLinesP(frame, 2, numpy.pi/180, 50,
                            numpy.array([]), minLineLength=10, maxLineGap=5)
    left_lane_line, right_lane_line = split_hough_lines(lines)
    left_lane_avg = numpy.average(left_lane_line, axis=0)
    right_lane_avg = numpy.average(right_lane_line, axis=0)
    left_coordinates = construct_common_line(
        left_lane_avg, original_image.shape)
    right_coordinates = construct_common_line(
        right_lane_avg, original_image.shape)
    cv2.line(original_image, (left_coordinates[0], left_coordinates[1]),
             (left_coordinates[2], left_coordinates[3]), (0, 255, 0), 10)
    cv2.line(original_image, (right_coordinates[0], right_coordinates[1]),
             (right_coordinates[2], right_coordinates[3]), (0, 255, 0), 10)
    return


def main():
    """
    Main method
    """
    image = cv2.imread("../data/image.jpg")
    processed_image = image_preprocessing(image)
    # ROI polygon
    custom_roi_polygon_points = numpy.array(
        [[264, image.shape[0]], [617, 409],
        [image.shape[1], image.shape[0]]])
    roi_image = extract_roi(processed_image, custom_roi_polygon_points)
    draw_hough_lines(image, roi_image)
    window_name = "lane_detection"
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    return


if __name__ == "__main__":
    main()
    pass
