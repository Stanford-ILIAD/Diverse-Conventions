#pragma once

#include <memory>

#include <madrona/python.hpp>

namespace Balance {

class Manager {
public:
    enum class ExecMode {
        CPU,
        CUDA,
    };
    
    struct Config {
        ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        bool debugCompile;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::Tensor doneTensor() const;
    MADRONA_IMPORT madrona::py::Tensor activeAgentTensor() const;
    MADRONA_IMPORT madrona::py::Tensor actionTensor() const;
    MADRONA_IMPORT madrona::py::Tensor observationTensor() const;
    MADRONA_IMPORT madrona::py::Tensor actionMaskTensor() const;
    MADRONA_IMPORT madrona::py::Tensor rewardTensor() const;

    MADRONA_IMPORT madrona::py::Tensor worldIDTensor() const;
    MADRONA_IMPORT madrona::py::Tensor agentIDTensor() const;


private:
    struct Impl;
    struct CPUImpl;

    #ifdef MADRONA_CUDA_SUPPORT
    struct GPUImpl;
    #endif

    std::unique_ptr<Impl> impl_;
};

}
