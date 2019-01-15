#pragma once 
#include "radx_core.hpp"
#include "radx_device.hpp"
#include "radx_shaders.hpp"

namespace radx {

    class Interface { // used for connection between algorithms and storage
    public:
        vuh::Array<uint32_t> referenceBuffer;
        vuh::Array<uint32_t> extraKeyBuffer;
    };

    class Algorithm : public std::enable_shared_from_this<Algorithm> {
        protected:
            std::shared_ptr<radx::Device> device;
            using Specs = vuh::typelist<uint32_t>;
            struct Params{uint32_t size; uint32_t bitfield;};

        public:
            std::unique_ptr<vuh::Program<Specs, Params>> histogramProgram;
            std::unique_ptr<vuh::Program<Specs, Params>> permuteProgram;
            std::unique_ptr<vuh::Program<Specs, Params>> workloadProgram;
            virtual std::shared_ptr<Algorithm> initialize(std::shared_ptr<radx::Device>& device);

            // can be used by children 
            virtual operator Algorithm&() { return *this; };
            virtual operator const Algorithm&() const { return *this; };
    };

    template <class T>
    class Sort : public std::enable_shared_from_this<Sort<T>> {
        protected:
            std::shared_ptr<T> algorithm;
            std::shared_ptr<radx::Device> device;
            std::shared_ptr<radx::Interface> interface;
            
            // side-load buffer
            std::shared_ptr<vuh::Array<uint32_t>> keys;
            std::shared_ptr<vuh::Array<uint32_t>> values;

        public:
        virtual VkResult initialize(std::shared_ptr<radx::Device>& device, std::shared_ptr<radx::Algorithm>& algorithm) {
            this->device = device;
            this->algorithm = std::dynamic_pointer_cast<T>(algorithm);
            return VK_SUCCESS;
        };

        virtual VkResult initialize(std::shared_ptr<radx::Device>& device, std::shared_ptr<T>& algorithm) {
            this->device = device;
            this->algorithm = algorithm;
            return VK_SUCCESS;
        };

        // for building arguments 
        virtual VkResult setKeys(std::shared_ptr<vuh::Array<uint32_t>> keys){ this->keys = keys; };
        virtual VkResult setValues(std::shared_ptr<vuh::Array<uint32_t>> values){ this->values = values; };

        // VUH does not support command buffers, but for future reserved command buffer reference acceptance 
        virtual VkResult buildCommand(){
            
        };

    };

    class Radix : public Algorithm, public std::enable_shared_from_this<Radix> {
        protected:
        public:
        virtual std::shared_ptr<Algorithm> initialize(std::shared_ptr<radx::Device>& device) override {
            return Algorithm::shared_from_this();
        };
    };

};
