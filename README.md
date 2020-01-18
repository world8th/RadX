# RadX (VK-1.1.130)

> You asked question, but how can be fast radix sort in NVIDIA Turing GPU's?

GPU sorting shaders dedication from vRt project. Optimized for modern GPU's and written on Vulkan API (GLSL).

## Tech Info 

```MD
- Architecture    : VKt (9:Gen)
- RHI             : Yes, Vulkan-HPP
- Support OS      : Windows 10 (2004 or Newer)
- Support HW      : RTX 2070 or familiar...
- Extensions      : NATIVE 
- Project Type    : Experimental, Personal
- Portfolio       : WIP
- Role/Target     : Faster and newer Radix Sort (8-bit and above)
- Graphical API   : Vulkan-API, Vulkan-HPP (compatible)
- Target Games    : Vulkan API Based...
- Code Language   : C++20, Visual Studio 2019 Support
- Commercial      : Under Consideration
- Execute Day     : [2020:Q1]
- Based On        : Vulkan.H, C++20, STL
- General To Do   : Complete radix-sort shaders for other GPU's
- Testing On GPU  : NVIDIA GeForce RTX 2070 (Gigabyte)

Date: 18.01.2020 helixd2k18@IMN
```

## What to be want to do

- Optimized sorting for NVIDIA RTX GPU's (and, probably, Volta GPU's)
- Remove outdated and bloated code
- Add new experimental features without rendering backend
- In future, add support for other possible architectures (Radeon VII, Navi, Ampere)
- Add support for Intel UHD Graphics 630 (if we will have time)
- CUDA Compute Cabability 7.5 Interporability

## Preview results

- In average can sort (up to, and more) 1 billion uint32_t elements per second (tested with RTX 2070, achievable in RTX 2070 Super)
- Outperform parallel std::sort up to 40x faster (Intel Core i7-8700K)
- Performance tested in Windows 10 (Insiders) and Visual Studio 2019 
- Can be built by GCC-8 in Linux systems (tested in Ubuntu 18.10)

## Materials 

- https://github.com/eloj/radix-sorting (basics of)
- http://www.heterogeneouscompute.org/wordpress/wp-content/uploads/2011/06/RadixSort.pdf (principles)
- https://vgc.poly.edu/~csilva/papers/cgf.pdf (partially)
- https://devblogs.nvidia.com/using-cuda-warp-level-primitives/ (planned real CUDA version)
- https://github.com/KhronosGroup/GLSL/blob/master/extensions/nv/GL_NV_shader_subgroup_partitioned.txt
- https://probablydance.com/2016/12/27/i-wrote-a-faster-sorting-algorithm/
- https://devblogs.nvidia.com/nvidia-turing-architecture-in-depth/ (optimal architecture)
