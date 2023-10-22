#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::py;

namespace Simplecooked {

using CPUExecutor =
    TaskGraphExecutor<Engine, Sim, Config, WorldInit>;
    
struct Manager::Impl {
    Config cfg;
    EpisodeManager *episodeMgr;

    static inline Impl * init(const Config &cfg);

    inline Impl(const Config &c, EpisodeManager *episode_mgr)
        : cfg(c),
          episodeMgr(episode_mgr)
    {}

    inline virtual ~Impl() {};
    virtual void run() = 0;
    virtual Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                Span<const int64_t> dims) = 0;
};

struct Manager::CPUImpl final : public Manager::Impl {
    CPUExecutor mwCPU;

    inline CPUImpl(const Config &cfg,
                   const Simplecooked::Config &app_cfg,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits,
                   uint32_t num_exported_buffers)
        : Impl(cfg, episode_mgr),
          mwCPU(ThreadPoolExecutor::Config {
                  .numWorlds = cfg.numWorlds,
                  .renderWidth = 0,
                  .renderHeight = 0,
                  .numExportedBuffers = num_exported_buffers,
                  .cameraMode = render::CameraMode::None,
                  .renderGPUID = 0,
              },
              app_cfg,
              world_inits)
    {}

    inline virtual ~CPUImpl() final
    {
        free(episodeMgr);
    }

    inline virtual void run() final { mwCPU.run(); }

    virtual inline Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = mwCPU.getExported(slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::GPUImpl final : public Manager::Impl {
    MWCudaExecutor mwGPU;

    inline GPUImpl(const Config &cfg,
                   const Simplecooked::Config &app_cfg,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits,
                   uint32_t num_exported_buffers)
        : Impl(cfg, episode_mgr),
          mwGPU({
                  .worldInitPtr = world_inits,
                  .numWorldInitBytes = sizeof(WorldInit),
                  .userConfigPtr = (void *)&app_cfg,
                  .numUserConfigBytes = sizeof(Simplecooked::Config),
                  .numWorldDataBytes = sizeof(Sim),
                  .worldDataAlignment = alignof(Sim),
                  .numWorlds = cfg.numWorlds,
                  .numExportedBuffers = num_exported_buffers, 
                  .gpuID = (uint32_t)cfg.gpuID,
                  .cameraMode = render::CameraMode::None,
                  .renderWidth = 0,
                  .renderHeight = 0,
              }, {
                  "",
                  { SIMPLECOOKED_SRC_LIST },
                  { SIMPLECOOKED_COMPILE_FLAGS },
                  cfg.debugCompile ? CompileConfig::OptMode::Debug :
                      CompileConfig::OptMode::LTO,
                  CompileConfig::Executor::TaskGraph,
              })
    {}

    inline virtual ~GPUImpl() final
    {
        REQ_CUDA(cudaFree(episodeMgr));
    }

    inline virtual void run() final { mwGPU.run(); }
    virtual inline Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = mwGPU.getExported(slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

Manager::Impl * Manager::Impl::init(const Config &cfg)
{
    EpisodeManager *episode_mgr;

    #ifdef MADRONA_CUDA_SUPPORT
    if (cfg.execMode == ExecMode::CPU ) {
        episode_mgr = (EpisodeManager *)malloc(sizeof(EpisodeManager));
        memset(episode_mgr, 0, sizeof(EpisodeManager));
    } else {
        episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));

        // Set the current episode count to 0
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));
    }
    #else
    episode_mgr = (EpisodeManager *)malloc(sizeof(EpisodeManager));
    memset(episode_mgr, 0, sizeof(EpisodeManager));
    #endif

    HeapArray<WorldInit> world_inits(cfg.numWorlds);

    Simplecooked::Config app_cfg {
                  // .terrain = cfg.terrain,
                  .height = cfg.height,
                  .width = cfg.width,
                  .num_players = cfg.num_players,
                  // .start_player_x = cfg.start_player_x,
                  // .start_player_y = cfg.start_player_y,
                  .placement_in_pot_rew = cfg.placement_in_pot_rew,
                  .dish_pickup_rew = cfg.dish_pickup_rew,
                  .soup_pickup_rew = cfg.soup_pickup_rew,
                  // .recipe_values = cfg.recipe_values,
                  // .recipe_times = cfg.recipe_times,
                  .horizon = cfg.horizon,
              };

    for (int r = 0; r < NUM_RECIPES; r++) {
        app_cfg.recipe_values[r] = cfg.recipe_values[r];
        app_cfg.recipe_times[r] = cfg.recipe_times[r];
    }

    for (int p = 0; p < cfg.num_players; p++) {
        app_cfg.start_player_x[p] = cfg.start_player_x[p];
        app_cfg.start_player_y[p] = cfg.start_player_y[p];
    }

    for (int x = 0; x < cfg.height * cfg.width; x++) {
        app_cfg.terrain[x] = (Simplecooked::TerrainT)cfg.terrain[x];
    }

    for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr,
        };
    }

    // Increase this number before exporting more tensors
    uint32_t num_exported_buffers = 10;

    if (cfg.execMode == ExecMode::CPU) {
        return new CPUImpl(cfg, app_cfg, episode_mgr, world_inits.data(),
            num_exported_buffers);
    } else {
        #ifdef MADRONA_CUDA_SUPPORT
        return new GPUImpl(cfg, app_cfg,
            episode_mgr, world_inits.data(), num_exported_buffers);
        #else
        return new CPUImpl(cfg, app_cfg, episode_mgr, world_inits.data(),
            num_exported_buffers);
        #endif
    }
}

MADRONA_EXPORT Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

MADRONA_EXPORT Manager::~Manager() {}

MADRONA_EXPORT void Manager::step()
{
    impl_->run();
}

MADRONA_EXPORT Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(0, Tensor::ElementType::Int32,
        {impl_->cfg.numWorlds});
}

MADRONA_EXPORT Tensor Manager::activeAgentTensor() const
{
    return impl_->exportTensor(1, Tensor::ElementType::Int32,
        {impl_->cfg.num_players, impl_->cfg.numWorlds});
}

MADRONA_EXPORT Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(2, Tensor::ElementType::Int32,
        {impl_->cfg.num_players, impl_->cfg.numWorlds, 1});
}

    // TODO
MADRONA_EXPORT Tensor Manager::observationTensor() const
{
    return impl_->exportTensor(3, Tensor::ElementType::Int8,
                               {impl_->cfg.num_players * impl_->cfg.width * impl_->cfg.height, impl_->cfg.numWorlds, sizeof(LocationXObservation)});
}

MADRONA_EXPORT Tensor Manager::actionMaskTensor() const
{
    return impl_->exportTensor(4, Tensor::ElementType::Int32,
        {impl_->cfg.num_players, impl_->cfg.numWorlds, NUM_MOVES});
}
    
MADRONA_EXPORT Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(5, Tensor::ElementType::Int32,
        {impl_->cfg.num_players, impl_->cfg.numWorlds});
}

MADRONA_EXPORT Tensor Manager::worldIDTensor() const
{
    return impl_->exportTensor(6, Tensor::ElementType::Int32,
        {impl_->cfg.num_players, impl_->cfg.numWorlds});
}

MADRONA_EXPORT Tensor Manager::agentIDTensor() const
{
    return impl_->exportTensor(7, Tensor::ElementType::Int32,
        {impl_->cfg.num_players, impl_->cfg.numWorlds});
}

MADRONA_EXPORT Tensor Manager::locationWorldIDTensor() const
{
    return impl_->exportTensor(8, Tensor::ElementType::Int32,
        {impl_->cfg.num_players * impl_->cfg.width * impl_->cfg.height, impl_->cfg.numWorlds});
}

MADRONA_EXPORT Tensor Manager::locationIDTensor() const
{
    return impl_->exportTensor(9, Tensor::ElementType::Int32,
        {impl_->cfg.num_players * impl_->cfg.width * impl_->cfg.height, impl_->cfg.numWorlds});
}

}
