//
// Created by duche on 2023/9/22.
//

#include "include/visionprocessapi.h"
#include "iostream"

using namespace std;

int main() {
    char* img_path = "C:\\Projects\\Delphi\\ImageShift\\Data\\253118ff-93ad-4c32-b03c-5a23a97251a6.jpg";
    float x = 0;
    float y = 0;
    int ret = CalculateMarkerCenter(img_path, &x, &y);
    if (ret != 0) {
        std::cout<<"calc error"<<ret<<endl;
    }

}