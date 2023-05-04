#pragma once

namespace Balance {

struct EpisodeManager {
    std::atomic_uint32_t curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
};

}
