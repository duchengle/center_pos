#include "opencv2\\highgui.hpp"
#include <opencv2\\imgproc.hpp>
#include <imgproc\\types_c.h>
#include <cstring>
#include "visionprocess.h"
#include "logger.h"

#define DEBUG_MODE    1
#define DEBUG_PROCESS    0

#ifndef GAUSSIAN_KERNEL_SIZE
#define GAUSSIAN_KERNEL_SIZE    5
#endif

#ifndef CANNY_THRESHOLD
#define CANNY_THRESHOLD            40
#endif
#ifndef CANNY_APERTURE_SIZE
#define CANNY_APERTURE_SIZE        3
#endif

#ifndef MIN_AREA_SIZE
#define MIN_AREA_SIZE            400
#endif

#ifndef MAX_PYRAMID_HIERARCHIES
#define MAX_PYRAMID_HIERARCHIES    2
#endif

#ifndef BOUNDARY_SIZE
#define BOUNDARY_SIZE            3
#endif

#ifndef EXIT_ROI_COUNT
#define EXIT_ROI_COUNT            2
#endif

CVisionProcessSolution::CVisionProcessSolution() {
    m_iErrNum = MY_PROCESS_OK;
    memset(m_strImagePath, 0, sizeof(m_strImagePath));
}

CVisionProcessSolution::~CVisionProcessSolution() = default;

INT32 CVisionProcessSolution::LoadImage(const CHAR *strImgPath) {
    add_log("2");
    add_log(strImgPath);
    if (!strImgPath) {
        m_iErrLine = __LINE__;
        m_iErrNum = MY_PROCESS_ERR_INVALID_PARAM;
        return m_iErrNum;
    }
    add_log("3");
    add_log(strImgPath);
    if (!m_matSrc.empty()) {
        m_matSrc.release();
    }
    add_log("4");
    add_log(strImgPath);
    strcpy_s(m_strImagePath, strImgPath);
    m_matSrc = cv::imread(strImgPath);
    if (m_matSrc.empty()) {
        add_log("5");
        add_log(strImgPath);
        m_iErrLine = __LINE__;
        m_iErrNum = MY_PROCESS_ERR_LOAD_IMAGE;
        return m_iErrNum;
    }

    return m_iErrNum;
}

INT32 CVisionProcessSolution::SaveResultImage(const CHAR *strImgPath) {
    cv::Mat matResultImage;
    INT32 iCrossSize = 0;

    if (m_matSrc.empty()) {
        m_iErrLine = __LINE__;
        m_iErrNum = MY_PROCESS_ERR_LOAD_IMAGE;
        return m_iErrNum;
    }
    if (3 <= m_matSrc.channels()) {
        m_matSrc.copyTo(matResultImage);
    } else {
        matResultImage = cv::Mat::zeros(m_matSrc.rows, m_matSrc.cols, CV_8UC3);
        std::vector<cv::Mat> channels;
        for (INT32 i = 0; i < 3; i++) {
            channels.push_back(m_matSrc);
        }
        merge(channels, matResultImage);
    }

    for (auto box : m_vectorContours) {
        ellipse(matResultImage, box, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        iCrossSize = std::min(box.size.height, box.size.width) / 2;
        cv::line(matResultImage, cv::Point(box.center.x - iCrossSize, box.center.y),
                 cv::Point(box.center.x + iCrossSize, box.center.y), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        cv::line(matResultImage, cv::Point(box.center.x, box.center.y - iCrossSize),
                 cv::Point(box.center.x, box.center.y + iCrossSize), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    cv::line(matResultImage,
             cv::Point(m_vectorContours[0].center.x, m_vectorContours[0].center.y),
             cv::Point(m_vectorContours[1].center.x, m_vectorContours[1].center.y),
             cv::Scalar(0.0, 255.0, 0.0),
             1,
             cv::LINE_8, 0);

    cv::line(matResultImage, cv::Point(m_ptCenter.x - iCrossSize, m_ptCenter.y),
             cv::Point(m_ptCenter.x + iCrossSize, m_ptCenter.y), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    cv::line(matResultImage, cv::Point(m_ptCenter.x, m_ptCenter.y - iCrossSize),
             cv::Point(m_ptCenter.x, m_ptCenter.y + iCrossSize), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

    cv::imwrite(strImgPath, matResultImage);

    return MY_PROCESS_OK;
}

INT32 CVisionProcessSolution::FindMarkerCenter(VOID) {
    if (!m_vectorContours.empty()) {
        m_vectorContours.clear();
    }

    m_iErrNum = FindContours(m_matSrc, m_vectorContours);

    return m_iErrNum;
}

INT32 CVisionProcessSolution::FindContours(const cv::Mat &matSrc, std::vector<cv::RotatedRect> &vectorContour) {
    cv::Mat matGray;

    if (matSrc.empty()) {
        m_iErrLine = __LINE__;
        m_iErrNum = MY_PROCESS_ERR_INVALID_PARAM;
        return m_iErrNum;
    }

    if (1 < matSrc.channels()) {
        cv::cvtColor(matSrc, matGray, CV_RGB2GRAY);
    } else {
        matGray = matSrc;
    }

    cv::Mat matPyramid = cv::Mat::zeros(matSrc.size(), CV_8UC3);
    m_iErrNum = FindContoursByPyramidLevel(matGray, matPyramid);
    if (MY_PROCESS_OK != m_iErrNum) {
        m_iErrLine = __LINE__;
        return m_iErrNum;
    }
//#if DEBUG_MODE && DEBUG_PROCESS
//    cv::Mat show;
//    cv::resize(matPyramid, show, cv::Size(matPyramid.cols / std::pow(2, 2), matPyramid.rows / std::pow(2, 2)));
//    cv::imshow("PyramidContour", show);
//#endif

    std::vector<std::vector<cv::Point>> vectorCandidateContours;
    std::vector<cv::Vec4i> vectorHierarchy;
    cv::Mat CircleImage(matPyramid.size(), CV_8UC1);
    cv::cvtColor(matPyramid, CircleImage, CV_RGB2GRAY);
    cv::findContours(CircleImage, vectorCandidateContours, vectorHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    if (vectorCandidateContours.size() < EXIT_ROI_COUNT) {
        m_iErrLine = __LINE__;
        m_iErrNum = MY_PROCESS_ERR_CONTOUR;
        return m_iErrNum;
    }

    for (INT32 iIdx = 0; iIdx < vectorCandidateContours.size(); iIdx++) {
        cv::RotatedRect box = cv::fitEllipse(vectorCandidateContours[iIdx]);
        if (abs(box.size.height / box.size.width - 1.0) < 0.1) {
            vectorContour.push_back(box);
        }
    }
    if (vectorContour.size() < EXIT_ROI_COUNT) {
        m_iErrLine = __LINE__;
        m_iErrNum = MY_PROCESS_ERR_CONTOUR;
        return m_iErrNum;
    }

    m_ptCenter.x = (vectorContour[0].center.x + vectorContour[1].center.x) / 2.0;
    m_ptCenter.y = (vectorContour[0].center.y + vectorContour[1].center.y) / 2.0;

#if DEBUG_MODE && DEBUG_PROCESS
    cv::Mat matCompareImage;
    matSrc.copyTo(matCompareImage);
    for (INT32 iIdx = 0; iIdx < vectorContour.size(); iIdx++)
    {
        cv::RotatedRect box = vectorContour[iIdx];
        ellipse(matCompareImage, box, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        INT32 iCrossSize = std::min(box.size.height, box.size.width) / 2;
        cv::line(matCompareImage, cv::Point(box.center.x - iCrossSize, box.center.y), cv::Point(box.center.x + iCrossSize, box.center.y), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::line(matCompareImage, cv::Point(box.center.x, box.center.y - iCrossSize), cv::Point(box.center.x, box.center.y + iCrossSize), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    cv::Mat result;
    cv::resize(matCompareImage, result, cv::Size(matCompareImage.cols / std::pow(2, 2), matCompareImage.rows / std::pow(2, 2)));
    cv::imshow("CompareDrawing", result);
    cv::waitKey(0);
#endif

    return MY_PROCESS_OK;
}

INT32 CVisionProcessSolution::FindContoursByPyramidLevel(cv::Mat matSrc, cv::Mat matContourImage) {
    INT32 iPyramidIdx = 0;
    std::vector<std::vector<cv::Point>> vectContours;
    std::vector<cv::Vec4i> vectHierarchy;
    std::vector<std::vector<cv::Point>> vectVilidContour;

    if (matSrc.empty()) {
        m_iErrLine = __LINE__;
        m_iErrNum = MY_PROCESS_ERR_INVALID_PARAM;
        return m_iErrNum;
    }

    for (iPyramidIdx = MAX_PYRAMID_HIERARCHIES; 0 < iPyramidIdx; iPyramidIdx--) {
        cv::Mat matPyramid;
        cv::resize(matSrc, matPyramid,
                   cv::Size(matSrc.cols / std::pow(2, iPyramidIdx), matSrc.rows / std::pow(2, iPyramidIdx)));
        cv::GaussianBlur(matPyramid, matPyramid, cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0, 0);
#if DEBUG_MODE && DEBUG_PROCESS
        cv::Mat pyramidshow;
        cv::resize(matPyramid, pyramidshow, cv::Size(matPyramid.cols / std::pow(2, 2), matPyramid.rows / std::pow(2, 2)));
        cv::imshow("Pyramid", pyramidshow);
#endif

        cv::Canny(matPyramid, matPyramid, CANNY_THRESHOLD, CANNY_THRESHOLD * 2, CANNY_APERTURE_SIZE);
#if DEBUG_MODE && DEBUG_PROCESS
        cv::Mat cannyshow;
        cv::resize(matPyramid, cannyshow, cv::Size(matPyramid.cols / std::pow(2, 2), matPyramid.rows / std::pow(2, 2)));
        cv::imshow("Canny", cannyshow);
#endif

        vectContours.clear();
        vectHierarchy.clear();
        cv::findContours(matPyramid, vectContours, vectHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

        vectVilidContour.clear();
        if (0 < vectContours.size()) {
            INT32 iMaxIdx = -1;
            DOUBLE dwArea = 0.0, dwMaxArea = (MIN_AREA_SIZE / pow(2, (iPyramidIdx - 1)));
            for (INT32 iIdx = 0; iIdx < vectContours.size(); iIdx++) {
                dwArea = cv::contourArea(vectContours[iIdx]);
                if (dwArea - (MIN_AREA_SIZE / pow(2, (iPyramidIdx - 1))) < 0.01) {
                    continue;
                } else {
                    vectVilidContour.push_back(vectContours[iIdx]);
                }
            }

            if (EXIT_ROI_COUNT <= vectVilidContour.size()) {
                for (INT32 iVectIdx = 0; iVectIdx < vectVilidContour.size(); iVectIdx++) {
                    for (INT32 iPtIdx = 0; iPtIdx < vectVilidContour[iVectIdx].size(); iPtIdx++) {
                        vectVilidContour[iVectIdx][iPtIdx].x *= pow(2, iPyramidIdx);
                        vectVilidContour[iVectIdx][iPtIdx].y *= pow(2, iPyramidIdx);
                    }
                    drawContours(matContourImage, vectVilidContour, iVectIdx, cv::Scalar(255, 255, 255), -1, 8,
                                 cv::Mat(), 0, cv::Point());
                }
                break;
            }
        }
    }

    if (0 == iPyramidIdx) {
        m_iErrLine = __LINE__;
        m_iErrNum = MY_PROCESS_ERR_CONTOUR;
    }

    return m_iErrNum;
}
