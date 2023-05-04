#pragma once

namespace Hanabi {

struct EpisodeManager {
    std::atomic_uint32_t curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;

    uint32_t colors;
    uint32_t ranks;
    uint32_t players;
    uint32_t max_information_tokens;
    uint32_t max_life_tokens;
};

}
