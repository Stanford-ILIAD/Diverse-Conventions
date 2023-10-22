#pragma once

namespace Simplecooked {

struct EpisodeManager {
    std::atomic_uint32_t curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
};

}
