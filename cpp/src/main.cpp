#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat denoiseImage(cv::Mat &image) {
    cv::Mat image_bw;
    if (image.type() != 0) {
        cv::cvtColor(image, image_bw, cv::COLOR_RGB2GRAY);
        image = std::move(image_bw);
    }
    cv::Mat blurred_image;
    cv::GaussianBlur(image, blurred_image, cv::Size(5,5), 0);
    return blurred_image;
}

cv::Mat detectEdge(const cv::Mat &image) {
    cv::Mat edge;
    cv::Canny(image, edge, 130, 240);
    return edge;
}

cv::Mat getROI(const cv::Mat &image, const std::vector<cv::Point2i> &roi_poly_points) {
    cv::Mat mask_image = cv::Mat::zeros(image.size(), image.type());
    cv::fillConvexPoly(mask_image, roi_poly_points, cv::Scalar(255));
    cv::Mat roi_image;
    cv::bitwise_and(image, mask_image, roi_image);
    return roi_image;
}

std::vector<cv::Vec4i> getHoughLines(const cv::Mat &image) {
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(image, lines, 2, CV_PI/180, 50, 10.0, 5.0);
    return lines;
}

std::vector<std::vector<cv::Vec2d>> seperateLines(const std::vector<cv::Vec4i> &lines,
                                                    const double slopeThresh = 0.3) {
    std::vector<std::vector<cv::Vec2d>> out_vec;
    out_vec.reserve(2);
    std::vector<cv::Vec2d> left_lane;
    left_lane.reserve(lines.size());
    std::vector<cv::Vec2d> right_lane;
    right_lane.reserve(lines.size());

    for (auto &line : lines) {
        double slope = static_cast<double>(line[1]-line[3])/static_cast<double>(line[0]-line[2]);
        if (std::abs(slope) < slopeThresh) {
            continue;
        }
        double intercept = static_cast<double>(line[1]) - (static_cast<double>(line[0])*slope);
        if (slope < 0) {
            left_lane.emplace_back(slope, intercept);
        } else {
            
            right_lane.emplace_back(slope, intercept);
        }
    }

    out_vec.emplace_back(left_lane);
    out_vec.emplace_back(right_lane);
    return out_vec;
}

cv::Vec4i getCommonLine(const std::vector<cv::Vec2d> &lane_lines,
                                     const int image_height) {
    auto avg_by_index = [](const std::vector<cv::Vec2d> &lane_lines, const int index) {
        double sum = 0;
        for (const auto &i : lane_lines) {
            sum += i[index];
        }
        return sum/lane_lines.size();
    };
    double avg_slope = avg_by_index(lane_lines, 0);
    double avg_intercept = avg_by_index(lane_lines, 1);
    int y1 = image_height;
    int y2 = static_cast<int>(y1 * (7.0/10.0));
    int x1 = static_cast<int>((y1 - avg_intercept)/avg_slope);
    int x2 = static_cast<int>((y2 - avg_intercept)/avg_slope);
    return {x1, y1, x2, y2};
}

void draw_image(const cv::Mat &image) {
    cv::imshow( "Display window", image);
    cv::waitKey(0);
}

void draw_image_with_lines(cv::Mat &image, const std::vector<cv::Vec4i> &lines) {
    for (auto &line_element : lines) {
        cv::line(image,
                {line_element[0], line_element[1]},
                {line_element[2], line_element[3]},
                cv::Scalar(0, 255, 0), 5, 16);
    }
    draw_image(image);
}

int main(int argc, char**argv) {
    cv::Mat image;
    cv::Mat original_image;
    cv::Mat image_gray;
    image = cv::imread("/home/shivang/my_projects/lane-detection/data/image.jpg", 1);
    image.copyTo(original_image);
    auto blurred_image = denoiseImage(image);
    auto canny = detectEdge(blurred_image);
    std::vector<cv::Point2i> roi_poly_points = {{264, canny.size().height},
                                                {617, 409},
                                                {canny.size().width, canny.size().height}};
    auto roi_image = getROI(canny, roi_poly_points);
    auto h_lines = getHoughLines(roi_image);
    auto seperated_lines = seperateLines(h_lines);
    auto left_line = getCommonLine(seperated_lines[0], roi_image.size().height);
    auto right_line = getCommonLine(seperated_lines[1], roi_image.size().height);
    std::vector<cv::Vec4i> final_lane_lines = {left_line, right_line};
    // draw_image(roi_image);
    draw_image_with_lines(original_image, final_lane_lines);
    
    return 0;
}