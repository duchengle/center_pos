//
// Created by duche on 2023/9/22.
//
#include "iostream"
#include "fstream"

void add_log(const char* msg) {
    std::ofstream file("example.txt");

    // 检查文件是否成功打开
    if (file.is_open()) {
        // 写入数据到文件
        file << msg << std::endl;

        // 关闭文件
        file.close();
    } else {
        std::cout << "Failed to open the file." << std::endl;
    }
}