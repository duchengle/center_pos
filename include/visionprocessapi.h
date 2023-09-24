#ifndef VISIONPROCESSAPI_H
#define VISIONPROCESSAPI_H

#define PLUGIN_API __declspec(dllexport)

extern "C" PLUGIN_API int CalculateMarkerCenter(const char *strImgPath, float *x, float *y);

#endif
