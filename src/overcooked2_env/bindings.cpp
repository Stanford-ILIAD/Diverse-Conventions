#include "mgr.hpp"

#include <madrona/macros.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#include <nanobind/stl/vector.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

using IntVector = std::vector<int64_t>;

namespace Simplecooked {

NB_MODULE(madrona_simplecooked_example_python, m) {
    nb::enum_<Manager::ExecMode>(m, "ExecMode")
        .value("CPU", Manager::ExecMode::CPU)
        .value("CUDA", Manager::ExecMode::CUDA)
        .export_values();
    nb::class_<Manager> (m, "SimplecookedSimulator")
        .def("__init__", [](Manager *self,
                            Manager::ExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,

                            IntVector terrain,
                            int64_t height,
                            int64_t width,
                            int64_t num_players,
                            IntVector start_player_x,
                            IntVector start_player_y,
                            int64_t placement_in_pot_rew,
                            int64_t dish_pickup_rew,
                            int64_t soup_pickup_rew,
                            IntVector recipe_values,
                            IntVector recipe_times,
                            int64_t horizon,
                            
                            bool debug_compile) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,

                .terrain = terrain,
                .height = height,
                .width = width,
                .num_players = num_players,
                .start_player_x = start_player_x,
                .start_player_y = start_player_y,
                .placement_in_pot_rew = placement_in_pot_rew,
                .dish_pickup_rew = dish_pickup_rew,
                .soup_pickup_rew = soup_pickup_rew,
                .recipe_values = recipe_values,
                .recipe_times = recipe_times,
                .horizon = horizon,
                
                .debugCompile = debug_compile,
            });
        },   nb::arg("exec_mode"),
             nb::arg("gpu_id"),
             nb::arg("num_worlds"),

             nb::arg("terrain"),
             nb::arg("height"),
             nb::arg("width"),
             nb::arg("num_players"),
             nb::arg("start_player_x"),
             nb::arg("start_player_y"),
             nb::arg("placement_in_pot_rew"),
             nb::arg("dish_pickup_rew"),
             nb::arg("soup_pickup_rew"),
             nb::arg("recipe_values"),
             nb::arg("recipe_times"),
             nb::arg("horizon"),
             
             nb::arg("debug_compile") = true)
        .def("step", &Manager::step)
        .def("done_tensor", &Manager::doneTensor)
        .def("active_agent_tensor", &Manager::activeAgentTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("agent_state_tensor", &Manager::observationTensor)
        .def("action_mask_tensor", &Manager::actionMaskTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("world_id_tensor", &Manager::worldIDTensor)
        .def("agent_id_tensor", &Manager::agentIDTensor)
        .def("location_world_id_tensor", &Manager::locationWorldIDTensor)
        .def("location_id_tensor", &Manager::locationIDTensor)
    ;
}

}
