#include "mgr.hpp"

#include <madrona/macros.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace Hanabi {

NB_MODULE(madrona_hanabi_example_python, m) {
    nb::enum_<Manager::ExecMode>(m, "ExecMode")
        .value("CPU", Manager::ExecMode::CPU)
        .value("CUDA", Manager::ExecMode::CUDA)
        .export_values();

    nb::class_<Manager> (m, "HanabiSimulator")
        .def("__init__", [](Manager *self,
                            Manager::ExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t colors,
                            int64_t ranks,
                            int64_t players,
                            int64_t max_information_tokens,
                            int64_t max_life_tokens,
                            bool debug_compile) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,

                .colors = (uint32_t)colors,
                .ranks = (uint32_t)ranks,
                .players = (uint32_t)players,
                .max_information_tokens = (uint32_t)max_information_tokens,
                .max_life_tokens = (uint32_t)max_life_tokens,
                
                .debugCompile = debug_compile,
            });
        }, nb::arg("exec_mode"), nb::arg("gpu_id"), nb::arg("num_worlds"),
           nb::arg("colors"), nb::arg("ranks"), nb::arg("players"),
           nb::arg("max_information_tokens"), nb::arg("max_life_tokens"),
           nb::arg("debug_compile") = true)
        .def("step", &Manager::step)
        .def("done_tensor", &Manager::doneTensor)
        .def("active_agent_tensor", &Manager::activeAgentTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("agent_state_tensor", &Manager::agentStateTensor)
        .def("action_mask_tensor", &Manager::actionMaskTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("world_id_tensor", &Manager::worldIDTensor)
        .def("agent_id_tensor", &Manager::agentIDTensor)
    ;
}

}
