#pragma once

// 
#include <fstream>
#include "vkt2/utils.hpp"
#include "vkt2/vector.hpp"

namespace radx {

    class Device;
    class PhysicalDeviceHelper;

    template<class T>
    class Sort;

    // radix sort algorithm
    class Algorithm;
    class Radix;
    //class VmaAllocatedBuffer;

    // 
    class InputInterface;
    class InternalInterface;

    // radix sort templated
    using RadixSort = Sort<Radix>;

};
