#pragma once 

// include vulkan API 
#include "radx_device.hpp"

namespace radx {
    
    class VmaAllocatedBuffer: public std::enable_shared_from_this<VmaAllocatedBuffer>  {
    public:
        VmaAllocatedBuffer();
        VmaAllocatedBuffer(
            const std::shared_ptr<radx::Device>& device, 
            vk::DeviceSize dsize = sizeof(uint32_t), 
            vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
            VmaMemoryUsage vmaUsage = VMA_MEMORY_USAGE_GPU_ONLY, bool alwaysMapped = false
        );

        // Get mapped memory
        void* map();

		// GPU unmap memory
        void unmap();

        // vk::Device caster
        //operator vk::Buffer&() { return buffer; };
        operator const vk::Buffer&() const { return buffer; };

        // Allocation
        //operator VmaAllocation&() { return allocation; };
        operator const VmaAllocation&() const { return allocation; };

        // AllocationInfo
        //operator VmaAllocationInfo&() { return allocationInfo; };
        operator const VmaAllocationInfo&() const { return allocationInfo; };
        
		// 
		operator const vk::DescriptorBufferInfo&() const { return bufInfo; };

    protected:
		void * mappedData = {};
        vk::Buffer buffer;
        VmaAllocation allocation;
        VmaAllocationInfo allocationInfo;
		VmaMemoryUsage usage = VMA_MEMORY_USAGE_GPU_ONLY;
        std::shared_ptr<radx::Device> device;
		vk::DescriptorBufferInfo bufInfo = {};
    };

	// TODO: buffer copying data and zero-initializer
	template<class T>
	class BufferRegion {
	public:
		BufferRegion(const std::shared_ptr<VmaAllocatedBuffer>& buffer, vk::DeviceSize size = 0ull, vk::DeviceSize offset = 0u) : buffer(buffer) {
			bufInfo.buffer = *buffer;
			bufInfo.offset = offset;
			bufInfo.range = size * sizeof(T);
			//this->map();
		};

		T* map() { mapped = (T*)((uint8_t*)buffer->map() + bufInfo.offset); return mapped; };
		void unmap() { buffer->unmap(); };

		T* data() { this->map(); return mapped; };
		T* data() const { return mapped; };
		size_t size() const { return size_t(bufInfo.range / sizeof(T)); };
		const vk::DeviceSize& range() const { return bufInfo.range; };

		// at function 
		const T& at(const uintptr_t& i) const { return mapped[i]; };
		T& at(const uintptr_t& i) { return mapped[i]; };

		// array operator 
		const T& operator [] (const uintptr_t& i) const { return at(i); };
		T& operator [] (const uintptr_t& i) { return at(i); };

		// begin ptr
		const T* begin() const { return &at(0); };
		T* begin() { return &at(0); };

		// end ptr
		const T* end() const { return &at(size() - 1ul); };
		T* end() { return &at(size() - 1ul); };

		operator const vk::DescriptorBufferInfo&() const { return bufInfo; };
		operator const vk::Buffer&() const { return *buffer; };
		const vk::DeviceSize& offset() const { return bufInfo.offset; };

	protected:
		T* mapped = {};
		std::shared_ptr<VmaAllocatedBuffer> buffer = {};
		vk::DescriptorBufferInfo bufInfo = {};
	};

	template<class T>
	class Vector {
	public:
		Vector() {}
		Vector(const std::shared_ptr<VmaAllocatedBuffer>& buffer, vk::DeviceSize size = 0ull, vk::DeviceSize offset = 0u) {
			region = std::make_shared<BufferRegion<T>>(buffer, size, offset);
		};
		Vector(const std::shared_ptr<BufferRegion<T>>& region) : region(region) {};
		Vector(const Vector<T>& vector) : region(vector.region) {};

		// map through
		T* map() { return region->map(); };
		void unmap() { return region->unmap(); };

		T* data() { return region->data(); };
		T* data() const { return region->data(); };

		// sizing 
		size_t size() const { return region->size(); };
		const vk::DeviceSize& range() const { return region->range(); };

		// at function 
		const T& at(const uintptr_t& i) const { return region->at(i); };
		T& at(const uintptr_t& i) { return region->at(i); };

		// array operator 
		const T& operator [] (const uintptr_t& i) const { at(i); };
		T& operator [] (const uintptr_t& i) { return at(i); };

		// begin ptr
		const T* begin() const { region->begin(); };
		T* begin() { return region->begin(); };

		// end ptr
		const T* end() const { return region->end(); };
		T* end() { return region->end(); };

		// 
		operator const vk::DescriptorBufferInfo&() const { return *region; };
		operator const vk::Buffer&() const { return *region; };
		const vk::DeviceSize& offset() const { return region->offset(); };

	protected:
		std::shared_ptr<BufferRegion<T>> region = {};
	};


};
