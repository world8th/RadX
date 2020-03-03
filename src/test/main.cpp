#define VKU_NO_GLFW

#if defined(_WIN32) && !defined(WIN32)
#define WIN32
#endif

#ifdef WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

//#ifdef OS_WIN
#if (defined(_WIN32) || defined(__MINGW32__) || defined(_MSC_VER_) || defined(__MINGW64__)) 
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#endif

//#ifdef OS_LNX
#ifdef __linux__
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#endif

//#define VMA_IMPLEMENTATION
#include "sort.hpp"

int main() {

    auto sorting = std::make_shared<rad::TestSort>();
    //system("pause");

};
