#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"
#include "rng.hpp"

// 3 cards for rank 0, 1 card for last, and 2 for other ranks

#define MAX_NUM_CARDS 50
#define HAND_SIZE 5
// max moves = discards (HAND_SIZE) + play (HAND_SIZE) + color (HAND_SIZE) + rank (HAND_SIZE)
#define NUM_MOVES 20

#define RANK 5
#define COLOR 5
#define N_PLAYERS 2
#define MAX_INFO_TOKENS 8
#define MAX_LIFE_TOKENS 3
#define ENCODE_HANDS_SIZE (RANK * COLOR * HAND_SIZE * (N_PLAYERS - 1) + N_PLAYERS)
#define ENCODE_ALL_HANDS_SIZE (RANK * COLOR * HAND_SIZE * N_PLAYERS + N_PLAYERS)
#define ENCODE_BOARD_SIZE (MAX_NUM_CARDS - N_PLAYERS * HAND_SIZE + COLOR * RANK + MAX_INFO_TOKENS + MAX_LIFE_TOKENS)
#define ENCODE_DISCARD_SIZE (MAX_NUM_CARDS)
#define ENCODE_LAST_ACTION_SIZE (N_PLAYERS * 2 + 4 + COLOR + RANK + 2 * HAND_SIZE + RANK * COLOR + 2)
#define V0_BELIEF_SIZE (N_PLAYERS * HAND_SIZE * (COLOR * RANK + COLOR + RANK))
#define OBS_SIZE (ENCODE_HANDS_SIZE + ENCODE_BOARD_SIZE + ENCODE_DISCARD_SIZE + ENCODE_LAST_ACTION_SIZE + V0_BELIEF_SIZE)
#define STATE_SIZE (ENCODE_ALL_HANDS_SIZE + ENCODE_BOARD_SIZE + ENCODE_DISCARD_SIZE + ENCODE_LAST_ACTION_SIZE + V0_BELIEF_SIZE)

namespace Hanabi {

    struct RendererInitStub {};

    // 3D Position & Quaternion Rotation
    // These classes are defined in madrona/components.hpp

    class Engine;

    // singletons
    
    struct WorldReset {
        int32_t resetNow;
    };

    struct Deck {
        uint8_t cards[MAX_NUM_CARDS];
        uint8_t size;
        // to reset, set cards[i], size=max_size
        // to pick next card, randomly swap some value from cards[0 to size-1]
        // with cards[size-1], and pick out that value
        uint8_t num_rem_cards[RANK * COLOR];
        
        uint8_t discard_counts[RANK * COLOR];
        // uint8_t discard_size;
        uint8_t fireworks[COLOR];
        uint8_t information_tokens;
        uint8_t life_tokens;

        uint8_t cur_player;

        int8_t turns_to_play;
        int8_t score;
        int8_t new_rew;
    };

        enum MoveType { kDiscard, kPlay, kRevealColor, kRevealRank, kInvalid };

    struct LastMove {
        MoveType move;
        int8_t player = -1;
        int8_t target_player = -1; // in absolute space
        int8_t card_index;
        bool scored = false;
        bool information_token = false;
        int8_t color = -1;
        int8_t rank = -1;
        uint8_t reveal_bitmask = 0;
        uint8_t newly_revealed_bitmask = 0;
        int8_t deal_to_player = -1;
    };

    // per-agent

    struct ActiveAgent {
        int32_t isActive;
    };

    struct Action {
        int32_t choice; // many discrete choices
    };
    
    struct Move {
        MoveType type;
        int8_t card_index;
        int8_t target_offset;
        int8_t color;
        int8_t rank;
    };

    struct Hand {
        uint8_t cards[HAND_SIZE];
        uint64_t card_plausible[HAND_SIZE];
        uint8_t size;
        int8_t known_color[HAND_SIZE];
        int8_t known_rank[HAND_SIZE];
        // card to index: color * num_ranks + rank
    };

    struct Observation {
        uint8_t bitvec[OBS_SIZE];
    };

    struct State {
        uint8_t bitvec[STATE_SIZE];
    };

    struct AgentID {
        int32_t id;
    };
    
    struct ActionMask {
        int32_t isValid[NUM_MOVES];
    };

    struct Reward {
        float rew;
    };

    // struct Agent : public madrona::Archetype<Action, Observation, Location, AgentID, ActionMask, ActiveAgent, Reward> {};

    struct Agent : public madrona::Archetype<Action, Observation, State, AgentID, ActionMask, ActiveAgent, Reward,
                                             Move, Hand> {};

    struct Config {
        uint32_t numPlayers;
    };

    struct Sim : public madrona::WorldBase {
        static void registerTypes(madrona::ECSRegistry &registry, const Config &cfg);

        static void setupTasks(madrona::TaskGraph::Builder &builder, const Config &cfg);

        Sim(Engine &ctx, const Config& cfg, const WorldInit &init);

        EpisodeManager *episodeMgr;
        RNG rng;

        uint32_t colors;
        uint32_t ranks;
        uint32_t players;
        uint32_t max_information_tokens;
        uint32_t max_life_tokens;

        uint32_t hand_size;

        madrona::Entity *agents;
    };

    class Engine : public ::madrona::CustomContext<Engine, Sim> {
        using CustomContext::CustomContext;
    };

}
