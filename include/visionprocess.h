#ifndef VISION_PROCESS_H_
#define VISION_PROCESS_H_

#include <opencv2\opencv.hpp>

#ifndef MY_PROCESS_OK
#define MY_PROCESS_OK					0
#endif

#ifndef MY_PROCESS_ERR_INVALID_PARAM
#define MY_PROCESS_ERR_INVALID_PARAM		0x1001
#endif
#ifndef MY_PROCESS_ERR_MEMORY
#define MY_PROCESS_ERR_MEMORY			0x1002
#endif
#ifndef MY_PROCESS_ERR_LOAD_IMAGE
#define MY_PROCESS_ERR_LOAD_IMAGE			0x1003
#endif
#ifndef MY_PROCESS_ERR_CONTOUR
#define MY_PROCESS_ERR_CONTOUR			0x1004
#endif

#ifndef MY_PROCESS_ERR_ROTATE_FIX
#define MY_PROCESS_ERR_ROTATE_FIX			0x1005
#endif

#ifndef VOID
typedef void		VOID;
#endif

#ifndef CHAR
typedef char		CHAR;
#endif

#ifndef INT32
typedef int			INT32;
#endif

#ifndef FLOAT
typedef float		FLOAT;
#endif

#ifndef DOUBLE
typedef double		DOUBLE;
#endif

class CVisionProcessSolution
{
public:
	CVisionProcessSolution();
	~CVisionProcessSolution();

public:
	VOID GetErrMsg(CHAR* pcMsgBuf, INT32 iBufLen) {
		sprintf_s(pcMsgBuf, iBufLen, "Process Error Local in %d Line, Error(%d)", m_iErrLine, m_iErrNum);
	}
	VOID GetMakerCenter(FLOAT *pfPtX, FLOAT *pfPtY) const {
		if (pfPtX)
			*pfPtX = m_ptCenter.x;
		if (pfPtY)
			*pfPtY = m_ptCenter.y;
	}

public:
	INT32 LoadImage(const CHAR *strImgPath);
	INT32 SaveResultImage(const CHAR* strImgPath);
	INT32 FindMarkerCenter(VOID);

private:
	INT32 FindContours(const cv::Mat& matSrc, std::vector<cv::RotatedRect>& vectorContour);
	INT32 FindContoursByPyramidLevel(cv::Mat matSrc, cv::Mat matContourImage);


public:

private:
	INT32 m_iErrLine{};
	INT32 m_iErrNum;
	CHAR m_strImagePath[512]{};
	cv::Mat m_matSrc;
	cv::Point m_ptCenter;
	std::vector<cv::RotatedRect> m_vectorContours;
};

#endif