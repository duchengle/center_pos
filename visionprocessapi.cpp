#include "visionprocessapi.h"
#include "visionprocess.h"
#include "include/logger.h"



int CalculateMarkerCenter(const char *strImgPath, float *x, float *y)
{
    int res = MY_PROCESS_OK;
    CVisionProcessSolution* pProcessObj;
    if ((!strImgPath) || (!x) || (!y))
    {
        return MY_PROCESS_ERR_INVALID_PARAM;
    }

    add_log("1");
    add_log(strImgPath);

    pProcessObj = new CVisionProcessSolution();
    res = pProcessObj->LoadImage(strImgPath);
    if (MY_PROCESS_OK != res)
    {
        delete pProcessObj;
        return res;
    }

    res = pProcessObj->FindMarkerCenter();
    if (MY_PROCESS_OK != res)
    {
        delete pProcessObj;
        return res;
    }

    pProcessObj->GetMakerCenter(x, y);
    delete pProcessObj;
    return res;
}