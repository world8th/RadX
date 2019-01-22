#pragma once 

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <future>
#include <fstream>

#include "radx_internal.hpp"
#include "radx_shaders.hpp"
#include "radx_utils.hpp"

namespace radx {

#ifdef RADX_IMPLEMENTATION
    VmaAllocatedBuffer::VmaAllocatedBuffer(
        const std::shared_ptr<radx::Device>& device, 
        vk::DeviceSize dsize, 
        vk::BufferUsageFlags bufferUsage, 
        VmaMemoryUsage vmaUsage, bool alwaysMapped
    ): device(device) {

        // Create the buffer object without memory.
        vk::BufferCreateInfo ci{};
        ci.size = dsize;
        ci.usage = bufferUsage;
        ci.sharingMode = vk::SharingMode::eExclusive;
        ci.queueFamilyIndexCount = device->queueFamilyIndices().size();
        ci.pQueueFamilyIndices = device->queueFamilyIndices().data();

        // 
        VmaAllocationCreateInfo aci{};
        aci.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
        aci.usage = this->usage = vmaUsage;

        //
        vmaCreateBuffer(*device, (VkBufferCreateInfo*)&ci, &aci, (VkBuffer*)&buffer, &allocation, &allocationInfo);

		// 
		bufInfo = { buffer, 0, VK_WHOLE_SIZE };
    };

    // Get mapped memory
    void* VmaAllocatedBuffer::map(){
        if (this->usage == VMA_MEMORY_USAGE_GPU_ONLY && !allocationInfo.pMappedData) {
            vmaMapMemory(*device, allocation, &mappedData);
        }
        else {
            mappedData = allocationInfo.pMappedData;
        };
        return mappedData;
    };

    // GPU unmap memory
    void VmaAllocatedBuffer::unmap(){ 
        if (this->usage == VMA_MEMORY_USAGE_GPU_ONLY && mappedData) {
            vmaUnmapMemory(*device, allocation);
        };
    };



    std::shared_ptr<Device> Device::initialize(const vk::Device& device, std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper){
        this->physicalHelper = physicalHelper;
        this->device = device;

        // get VMA allocator for device
        if (this->physicalHelper->allocator) 
        {
            this->allocator = this->physicalHelper->allocator;
        };

        
        if (!this->allocator) 
        {
            // load API calls for context
            volkLoadDevice(vk::Device(*this));

            // create VMA memory allocator (with Volk support)
#ifdef VOLK_H_
            VolkDeviceTable vktable;
            volkLoadDeviceTable(&vktable, vk::Device(*this));

            // VMA functions with Volk compatibility
            VmaVulkanFunctions vfuncs = {};
            vfuncs.vkAllocateMemory = vktable.vkAllocateMemory;
            vfuncs.vkBindBufferMemory = vktable.vkBindBufferMemory;
            vfuncs.vkBindImageMemory = vktable.vkBindImageMemory;
            vfuncs.vkCreateBuffer = vktable.vkCreateBuffer;
            vfuncs.vkCreateImage = vktable.vkCreateImage;
            vfuncs.vkDestroyBuffer = vktable.vkDestroyBuffer;
            vfuncs.vkDestroyImage = vktable.vkDestroyImage;
            vfuncs.vkFreeMemory = vktable.vkFreeMemory;
            vfuncs.vkGetBufferMemoryRequirements = vktable.vkGetBufferMemoryRequirements;
            vfuncs.vkGetBufferMemoryRequirements2KHR = vktable.vkGetBufferMemoryRequirements2KHR;
            vfuncs.vkGetImageMemoryRequirements = vktable.vkGetImageMemoryRequirements;
            vfuncs.vkGetImageMemoryRequirements2KHR = vktable.vkGetImageMemoryRequirements2KHR;
            vfuncs.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
            vfuncs.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
            vfuncs.vkMapMemory = vktable.vkMapMemory;
            vfuncs.vkUnmapMemory = vktable.vkUnmapMemory;
            vfuncs.vkInvalidateMappedMemoryRanges = vktable.vkInvalidateMappedMemoryRanges;
#endif
            
            // create Vma allocator
            VmaAllocatorCreateInfo allocatorInfo = {};
#ifdef VOLK_H_
            allocatorInfo.pVulkanFunctions = &vfuncs;
#endif
            allocatorInfo.physicalDevice = vk::PhysicalDevice(*this->physicalHelper);
            allocatorInfo.device = vk::Device(*this);
            allocatorInfo.preferredLargeHeapBlockSize = 16 * sizeof(uint32_t);
            allocatorInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT || VMA_ALLOCATION_CREATE_MAPPED_BIT;
            allocatorInfo.pAllocationCallbacks = nullptr;
            allocatorInfo.pHeapSizeLimit = nullptr;
            vmaCreateAllocator(&allocatorInfo, &this->allocator);
        };

        // descriptor pool
        if (!this->descriptorPool) 
        {
            // pool sizes, and create descriptor pool
            std::vector<vk::DescriptorPoolSize> psizes = { };
            psizes.push_back(vk::DescriptorPoolSize().setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(128));
            psizes.push_back(vk::DescriptorPoolSize().setType(vk::DescriptorType::eStorageTexelBuffer).setDescriptorCount(128));
            psizes.push_back(vk::DescriptorPoolSize().setType(vk::DescriptorType::eInlineUniformBlockEXT).setDescriptorCount(128));
            psizes.push_back(vk::DescriptorPoolSize().setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(128));

            vk::DescriptorPoolInlineUniformBlockCreateInfoEXT inlineDescPool{};
            inlineDescPool.maxInlineUniformBlockBindings = 2;
            this->descriptorPool = vk::Device(*this).createDescriptorPool(vk::DescriptorPoolCreateInfo().setPNext(&inlineDescPool).setPPoolSizes(psizes.data()).setPoolSizeCount(psizes.size()).setMaxSets(256).setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBindEXT));
        };

        // pipeline cache 
        if (!this->pipelineCache) 
        {
            this->pipelineCache = vk::Device(*this).createPipelineCache(vk::PipelineCacheCreateInfo());
        };

        { // create descriptor layouts 
            const auto pbindings = vk::DescriptorBindingFlagBitsEXT::ePartiallyBound | vk::DescriptorBindingFlagBitsEXT::eUpdateAfterBind | vk::DescriptorBindingFlagBitsEXT::eVariableDescriptorCount | vk::DescriptorBindingFlagBitsEXT::eUpdateUnusedWhilePending;
            const auto vkfl = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT().setPBindingFlags(&pbindings);
            const auto vkpi = vk::DescriptorSetLayoutCreateInfo().setPNext(&vkfl);

            {
                const std::vector<vk::DescriptorSetLayoutBinding> _bindings = {
                    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // keys cache
                    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // values cache
                    vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // radice cache
                    vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // histogram of radices (every work group)
                    vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // prefix-sum of radices (every work group)
                    vk::DescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
                    vk::DescriptorSetLayoutBinding(6, vk::DescriptorType::eInlineUniformBlockEXT, sizeof(uint32_t), vk::ShaderStageFlagBits::eCompute) // inline uniform data of algorithms
                };

                //const std::vector<vk::DescriptorBindingFlagsEXT> _bindingFlags = { {}, {}, {}, {}, {}, vk::DescriptorBindingFlagBitsEXT::ePartiallyBound };
                //const auto vkfl = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT().setPBindingFlags(_bindingFlags.data()).setBindingCount(_bindingFlags.size());
                descriptorLayouts.push_back(device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(vkpi).setPBindings(_bindings.data()).setBindingCount(_bindings.size())));
            };

            {
                const std::vector<vk::DescriptorSetLayoutBinding> _bindings = {
                    vk::DescriptorSetLayoutBinding(0 , vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // keys in
                    vk::DescriptorSetLayoutBinding(1 , vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // values in
                    vk::DescriptorSetLayoutBinding(6 , vk::DescriptorType::eInlineUniformBlockEXT, sizeof(uint32_t), vk::ShaderStageFlagBits::eCompute)
                };
                //const std::vector<vk::DescriptorBindingFlagsEXT> _bindingFlags = { {}, {}, vk::DescriptorBindingFlagBitsEXT::ePartiallyBound };
                //const auto vkfl = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT().setPBindingFlags(_bindingFlags.data()).setBindingCount(_bindingFlags.size());
                descriptorLayouts.push_back(device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(vkpi).setPBindings(_bindings.data()).setBindingCount(_bindings.size())));
            };
        };

        return shared_from_this();
    };



     InternalInterface& InternalInterface::buildDescriptorSet(){
        std::vector<vk::DescriptorSetLayout> dsLayouts = { device->getDescriptorSetLayoutSupport().at(0) };
        this->descriptorSet = vk::Device(*device).allocateDescriptorSets(vk::DescriptorSetAllocateInfo().setDescriptorPool(*device).setPSetLayouts(&dsLayouts[0]).setDescriptorSetCount(1)).at(0);

        // if no has buffer, set it!
        if (!this->keysStoreBufferInfo.buffer) this->keysStoreBufferInfo.buffer = *bufferMemory;
        if (!this->valuesStoreBufferInfo.buffer) this->valuesStoreBufferInfo.buffer = *bufferMemory;
        if (!this->keysCacheBufferInfo.buffer) this->keysCacheBufferInfo.buffer = *bufferMemory;
        if (!this->histogramBufferInfo.buffer) this->histogramBufferInfo.buffer = *bufferMemory;
        if (!this->prefixScansBufferInfo.buffer) this->prefixScansBufferInfo.buffer = *bufferMemory;
        if (!this->referencesBufferInfo.buffer) this->referencesBufferInfo.buffer = *bufferMemory;

        // write into descriptor set
        const auto writeTmpl = vk::WriteDescriptorSet(this->descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer);
        std::vector<vk::WriteDescriptorSet> writes = {
            vk::WriteDescriptorSet(writeTmpl).setDstBinding(0).setPBufferInfo(&this->keysStoreBufferInfo),
            vk::WriteDescriptorSet(writeTmpl).setDstBinding(1).setPBufferInfo(&this->valuesStoreBufferInfo),
            vk::WriteDescriptorSet(writeTmpl).setDstBinding(2).setPBufferInfo(&this->keysCacheBufferInfo),
            vk::WriteDescriptorSet(writeTmpl).setDstBinding(3).setPBufferInfo(&this->histogramBufferInfo),
            vk::WriteDescriptorSet(writeTmpl).setDstBinding(4).setPBufferInfo(&this->prefixScansBufferInfo),
            vk::WriteDescriptorSet(writeTmpl).setDstBinding(5).setPBufferInfo(&this->referencesBufferInfo),
        };

        // inline descriptor 
        vk::WriteDescriptorSetInlineUniformBlockEXT inlineDescriptorData{};
        {
            inlineDescriptorData.dataSize = sizeof(uint32_t);
            inlineDescriptorData.pData = &this->maxElementCount;
            writes.push_back(vk::WriteDescriptorSet(writeTmpl).setDstBinding(6).setDescriptorCount(inlineDescriptorData.dataSize).setDescriptorType(vk::DescriptorType::eInlineUniformBlockEXT).setPNext(&inlineDescriptorData));
        };

        vk::Device(*this->device).updateDescriptorSets(writes, {});
        return *this;
    };


     InputInterface& InputInterface::buildDescriptorSet(){
        std::vector<vk::DescriptorSetLayout> dsLayouts = { device->getDescriptorSetLayoutSupport().at(1) };
        this->descriptorSet = vk::Device(*device).allocateDescriptorSets(vk::DescriptorSetAllocateInfo().setDescriptorPool(*device).setPSetLayouts(&dsLayouts[0]).setDescriptorSetCount(1)).at(0);

        // input data 
        const auto writeTmpl = vk::WriteDescriptorSet(this->descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer);
        std::vector<vk::WriteDescriptorSet> writes = {
            vk::WriteDescriptorSet(writeTmpl).setDstBinding(0).setPBufferInfo(&this->keysBufferInfo),
            vk::WriteDescriptorSet(writeTmpl).setDstBinding(1).setPBufferInfo(&this->valuesBufferInfo),
        };

        // inline descriptor 
        vk::WriteDescriptorSetInlineUniformBlockEXT inlineDescriptorData{};
        {
            inlineDescriptorData.dataSize = sizeof(uint32_t);
            inlineDescriptorData.pData = &this->elementCount;
            writes.push_back(vk::WriteDescriptorSet(writeTmpl).setDstBinding(6).setDescriptorCount(inlineDescriptorData.dataSize).setDescriptorType(vk::DescriptorType::eInlineUniformBlockEXT).setPNext(&inlineDescriptorData));
        };

        vk::Device(*this->device).updateDescriptorSets(writes, {});
        return *this;
    };


    VkResult Radix::initialize(const std::shared_ptr<radx::Device>& device) {
        this->device = device;
        std::vector<vk::DescriptorSetLayout> setLayouts = device->getDescriptorSetLayoutSupport();

        // push constant ranges
        vk::PushConstantRange pConstRange{};
        pConstRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pConstRange.offset = 0u;
        pConstRange.size = sizeof(uint32_t)*4u;

        // pipeline layout create info
        vk::PipelineLayoutCreateInfo pplLayoutCi{};
        pplLayoutCi.setLayoutCount = setLayouts.size();
        pplLayoutCi.pSetLayouts = setLayouts.data();
        pplLayoutCi.pushConstantRangeCount = 1;
        pplLayoutCi.pPushConstantRanges = &pConstRange;

        // create pipeline layout 
        this->pipelineLayout = vk::Device(*device).createPipelineLayout(pplLayoutCi);
        this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::histogram, *device->getPhysicalHelper()), pipelineLayout, *device));
        this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::workload, *device->getPhysicalHelper()), pipelineLayout, *device));
        this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::permute, *device->getPhysicalHelper()), pipelineLayout, *device));
        this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::copyhack, *device->getPhysicalHelper()), pipelineLayout, *device));
        this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::transposer, *device->getPhysicalHelper()), pipelineLayout, *device));

        // return shared_ptr when needed
        return VK_SUCCESS;
    };


    VkResult Radix::command(const vk::CommandBuffer& cmdBuf, const std::unique_ptr<radx::InternalInterface>& internalInterface, const std::shared_ptr<radx::InputInterface>& inputInterface, VkResult& vkres) {
        std::vector<vk::DescriptorSet> descriptors = {*internalInterface, *inputInterface};
        cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->pipelineLayout, 0, descriptors, {});
        commandBarrier(cmdBuf);

		const uint32_t stageCount = radx::Vendor(*device->getPhysicalHelper()) == radx::Vendor::NV_TURING ? 4u : 8u;
        for (auto I=0u;I<stageCount;I++) { // TODO: add support variable stage length

            std::array<uint32_t,4> stageC = {I,0,0,0};
            cmdBuf.pushConstants(this->pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0u, sizeof(uint32_t)*4, &stageC[0]);
            commandBarrier(cmdBuf);

            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->copyhack]);
            cmdBuf.dispatch(this->groupX, 1u, 1u);
            commandBarrier(cmdBuf);

            // still broken
            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->transposer]);
            cmdBuf.dispatch(this->groupX, 1u, 1u);
            commandBarrier(cmdBuf);

            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->histogram]);
            cmdBuf.dispatch(this->groupX, 1u, 1u);
            commandBarrier(cmdBuf);

            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->workload]);
            cmdBuf.dispatch(1u, 1u, 1u);
            commandBarrier(cmdBuf);

            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->permute]);
            cmdBuf.dispatch(this->groupX, 1u, 1u);
            commandBarrier(cmdBuf);

        };

        return VK_SUCCESS;
    };


    VkResult Radix::createInternalMemory(std::unique_ptr<radx::InternalInterface>& internalInterface, const size_t& maxElementCount) {
        vk::DeviceSize tileFix = 4u*this->groupX;

        vk::DeviceSize 
            inlineSize = 0,//sizeof(uint32_t) * 4ull,
            keysSize = maxElementCount * sizeof(uint32_t),
            valuesSize = maxElementCount * sizeof(uint32_t),
            keyCacheSize = tiled(maxElementCount, tileFix) * tileFix * sizeof(uint8_t),
            referencesSize = 16ull * sizeof(uint32_t),//maxElementCount * sizeof(uint32_t),
            histogramsSize = 256ull * (this->groupX+1) * sizeof(uint32_t),
            prefixScanSize = histogramsSize
            ;

        vk::DeviceSize 
            keysOffset = inlineSize,
            valuesOffset = keysOffset + keysSize,
            keyCacheOffset = valuesOffset + valuesSize,
            referencesOffset = keyCacheOffset + keyCacheSize,
            histogramsOffset = referencesOffset + referencesSize,
            prefixScanOffset = histogramsOffset + histogramsSize
            ;

        // get memory size and set max element count
        vk::DeviceSize memorySize = prefixScanOffset + prefixScanSize;
        internalInterface->setMaxElementCount(maxElementCount);

        // on deprecation 
        internalInterface->setKeysStoreBufferInfo(vk::DescriptorBufferInfo(nullptr, keysOffset, keysSize));
        internalInterface->setValuesStoreBufferInfo(vk::DescriptorBufferInfo(nullptr, valuesOffset, valuesSize));

        // next-gen featured buffers
        internalInterface->setKeysCacheBufferInfo(vk::DescriptorBufferInfo(nullptr, keyCacheOffset, keyCacheSize));
        internalInterface->setReferencesBufferInfo(vk::DescriptorBufferInfo(nullptr, referencesOffset, referencesSize));

        // still required for effective sorting 
        internalInterface->setHistogramBufferInfo(vk::DescriptorBufferInfo(nullptr, histogramsOffset, histogramsSize));
        internalInterface->setPrefixScansBufferInfo(vk::DescriptorBufferInfo(nullptr, prefixScanOffset, prefixScanSize));

        // command for build descriptor set
        internalInterface->buildMemory(memorySize).buildDescriptorSet();

        return VK_SUCCESS;
    };
#endif

};
