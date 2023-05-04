#pragma once

#include <memory>
#include <vector>

#include <madrona/python.hpp>

using IntVector = std::vector<int64_t>;

namespace Simplecooked {

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
        
        IntVector terrain;
        int64_t height;
        int64_t width;
        int64_t num_players;
        IntVector start_player_x;
        IntVector start_player_y;
        int64_t placement_in_pot_rew;
        int64_t dish_pickup_rew;
        int64_t soup_pickup_rew;
        IntVector recipe_values;
        IntVector recipe_times;
        int64_t horizon;
        
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
    MADRONA_IMPORT madrona::py::Tensor locationWorldIDTensor() const;
    MADRONA_IMPORT madrona::py::Tensor locationIDTensor() const;


private:
    struct Impl;
    struct CPUImpl;

    #ifdef MADRONA_CUDA_SUPPORT
    struct GPUImpl;
    #endif

    std::unique_ptr<Impl> impl_;
};

}
