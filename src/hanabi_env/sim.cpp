#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

#include<cmath>

using namespace madrona;
using namespace madrona::math;

namespace Hanabi {

    
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<Deck>();
    registry.registerSingleton<LastMove>();
    
    registry.registerComponent<Action>();
    registry.registerComponent<Observation>();
    registry.registerComponent<State>();
    registry.registerComponent<AgentID>();
    registry.registerComponent<ActionMask>();
    registry.registerComponent<ActiveAgent>();
    registry.registerComponent<Reward>();
    
    registry.registerComponent<Move>();
    registry.registerComponent<Hand>();

    registry.registerFixedSizeArchetype<Agent>(cfg.numPlayers);

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, ActiveAgent>(1);
    registry.exportColumn<Agent, Action>(2);
    registry.exportColumn<Agent, Observation>(3);
    registry.exportColumn<Agent, ActionMask>(4);
    registry.exportColumn<Agent, Reward>(5);
    registry.exportColumn<Agent, WorldID>(6);
    registry.exportColumn<Agent, AgentID>(7);
    registry.exportColumn<Agent, State>(8);
}

    inline uint8_t drawDeck(Engine &ctx, Deck &deck)
{
    int32_t swaploc = static_cast<int32_t>(deck.size * ctx.data().rng.rand());
    uint8_t retval = deck.cards[swaploc];
    deck.cards[swaploc] = deck.cards[deck.size - 1];
    deck.size--;
    return retval;
}

    inline int encodeHands(Engine &ctx, Entity &agent, int offset)
{
    Observation &obs = ctx.getUnsafe<Observation>(agent);
    // Observation &state = ctx.getUnsafe<State>(agent);
    int32_t agent_id = ctx.getUnsafe<AgentID>(agent).id;

    uint32_t num_players = ctx.data().players;
    int bits_per_card = ctx.data().colors * ctx.data().ranks;
    for (int i = 1; i < num_players; i++) {
        int partner_id = (agent_id + i) % num_players;
        Entity partner_agent = ctx.data().agents[partner_id];

        Hand &partner_hand = ctx.getUnsafe<Hand>(partner_agent);
        for (int cardnum = 0; cardnum < partner_hand.size; cardnum++) {
            uint8_t actual_id = partner_hand.cards[cardnum];
            for (int b = 0; b < bits_per_card; b++) {
                obs.bitvec[offset++] = (b == actual_id);
            }
        }

        for (int cardnum = partner_hand.size; cardnum < ctx.data().hand_size; cardnum++) {
            for (int b = 0; b < bits_per_card; b++) {
                obs.bitvec[offset++] = 0;
            }
        }

    }

    for (int i = 0; i < num_players; i++) {
        int partner_id = (agent_id + i) % num_players;
        Entity partner_agent = ctx.data().agents[partner_id];

        obs.bitvec[offset++] = (ctx.getUnsafe<Hand>(partner_agent).size < ctx.data().hand_size);

    }
    return offset;
}

    inline int encodeBoard(Engine &ctx, Entity &agent, int offset)
{
    Observation &obs = ctx.getUnsafe<Observation>(agent);
    // Observation &state = ctx.getUnsafe<State>(agent);
    Deck &deck = ctx.getSingleton<Deck>();
    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;

    // deck size
    for (int i = 0; i < deck.size; i++) {
        obs.bitvec[offset++] = 1;
    }

    int max_size = ((4 + (ranks - 2) * 2) * colors - ctx.data().hand_size * ctx.data().players);
    
    for (int i = deck.size; i < max_size; i++) {
        obs.bitvec[offset++] = 0;
    }

    // fireworks
    for (int c = 0; c < colors; c++) {
        for (int i = 0; i < ranks; i++) {
            obs.bitvec[offset++] = (i + 1 == deck.fireworks[c]);
        }
    }

    // info tokens
    for (int i = 0; i < deck.information_tokens; i++) {
        obs.bitvec[offset++] = 1;
    }
    for (int i = deck.information_tokens; i < ctx.data().max_information_tokens; i++) {
        obs.bitvec[offset++] = 0;
    }

    // life tokens
    for (int i = 0; i < deck.life_tokens; i++) {
        obs.bitvec[offset++] = 1;
    }
    for (int i = deck.life_tokens; i < ctx.data().max_life_tokens; i++) {
        obs.bitvec[offset++] = 0;
    }
    
    return offset;
}

    inline int encodeDiscards(Engine &ctx, Entity &agent, int offset)
{
    Observation &obs = ctx.getUnsafe<Observation>(agent);
    // Observation &state = ctx.getUnsafe<State>(agent);
    Deck &deck = ctx.getSingleton<Deck>();
    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;
    
    int32_t id = 0;
    for (int c = 0; c < colors; c++) {
        for (int r = 0; r < ranks; r++) {
            int cr_num = (r == 0 ? 3 : r == ranks - 1 ? 1 : 2);
            for (int i = 0; i < cr_num; i++) {
                obs.bitvec[offset++] = (deck.discard_counts[id] > i);
            }
            id++;
        }
    }
    
    return offset;
}

    inline int encodeLastAction(Engine &ctx, Entity &agent, int offset)
{    
    Observation &obs = ctx.getUnsafe<Observation>(agent);
    // Observation &state = ctx.getUnsafe<State>(agent);
    LastMove &lastmove = ctx.getSingleton<LastMove>();
    
    int32_t agent_id = ctx.getUnsafe<AgentID>(agent).id;

    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;
    int32_t num_players = ctx.data().players;
    uint32_t hand_size = ctx.data().hand_size;

    int32_t relative_agent = (lastmove.player == -1 ? -1 : (agent_id - lastmove.player + num_players) % num_players);
    
    for (int i = 0; i < ctx.data().players; i++) {
        obs.bitvec[offset++] = (i == relative_agent);
    }

    for (int i = 0; i < 4; i++) {
        obs.bitvec[offset + i] = 0;
    }

    switch (lastmove.move) {
      case MoveType::kPlay:
        obs.bitvec[offset] = 1;
        break;
      case MoveType::kDiscard:
        obs.bitvec[offset + 1] = 1;
        break;
      case MoveType::kRevealColor:
        obs.bitvec[offset + 2] = 1;
        break;
      case MoveType::kRevealRank:
        obs.bitvec[offset + 3] = 1;
        break;
    }
    offset += 4;

    if (lastmove.move == MoveType::kRevealColor ||
        lastmove.move == MoveType::kRevealRank) {
        int8_t observer_relative_target = (agent_id - lastmove.target_player + num_players) % num_players;
        for (int i = 0; i < num_players; i++) {
            obs.bitvec[offset+i] = (i == observer_relative_target);
        }
    } else {
        for (int i = 0; i < num_players; i++) {
            obs.bitvec[offset+i] = 0;
        }
    }

    offset += num_players;

    
    if (lastmove.move == MoveType::kRevealColor) {
        int8_t chosen_color = lastmove.color;
        for (int i = 0; i < colors; i++) {
            obs.bitvec[offset+i] = (i == chosen_color);
        }
    } else {
        for (int i = 0; i < colors; i++) {
            obs.bitvec[offset+i] = 0;
        }
    }

    offset += colors;

    if (lastmove.move == MoveType::kRevealRank) {
        int8_t chosen_rank = lastmove.rank;
        for (int i = 0; i < ranks; i++) {
            obs.bitvec[offset+i] = (i == chosen_rank);
        }
    } else {
        for (int i = 0; i < ranks; i++) {
            obs.bitvec[offset+i] = 0;
        }
    }

    offset += ranks;

    if (lastmove.move == MoveType::kRevealColor ||
        lastmove.move == MoveType::kRevealRank) {
        for (int i = 0, mask = 1; i < hand_size; ++i, mask <<= 1) {
            obs.bitvec[offset+i] = ((lastmove.reveal_bitmask & mask) > 0);
        }
    } else {
        for (int i = 0, mask = 1; i < hand_size; ++i, mask <<= 1) {
            obs.bitvec[offset+i] = 0;
        }
    }

    offset += hand_size;

    if (lastmove.move == MoveType::kPlay ||
        lastmove.move == MoveType::kDiscard) {
        for (int i = 0; i < hand_size; i++) {
            obs.bitvec[offset+i] = (i == lastmove.card_index);
        }
    } else {
        for (int i = 0; i < hand_size; i++) {
            obs.bitvec[offset+i] = 0;
        }
    }

    offset += hand_size;
    
    if (lastmove.move == MoveType::kPlay ||
        lastmove.move == MoveType::kDiscard) {
        for (int i = 0; i < colors * ranks; i++) {
            obs.bitvec[offset+i] = (i == (lastmove.color * ranks + lastmove.rank));
        }
    } else {
        for (int i = 0; i < colors * ranks; i++) {
            obs.bitvec[offset+i] = 0;
        }
    }

    offset += colors * ranks;


    if (lastmove.move == MoveType::kPlay) {
        obs.bitvec[offset] = lastmove.scored;
        obs.bitvec[offset+1] = lastmove.information_token;
    } else {
        obs.bitvec[offset] = 0;
        obs.bitvec[offset+1] = 0;
    }

    offset += 2;
    
    return offset;
}

    inline int encodeCardKnowledge(Engine &ctx, Entity &agent, int offset)
{
    Observation &obs = ctx.getUnsafe<Observation>(agent);
    // Observation &state = ctx.getUnsafe<State>(agent);
    Deck &deck = ctx.getSingleton<Deck>();
    int32_t agent_id = ctx.getUnsafe<AgentID>(agent).id;
    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;
    uint32_t num_players = ctx.data().players;
    uint32_t hand_size = ctx.data().hand_size;
    uint32_t bits_per_card = colors * ranks;

    for (int i = 0; i < num_players; i++) {
        int partner_id = (agent_id + i) % num_players;
        Entity partner_agent = ctx.data().agents[partner_id];

        Hand &partner_hand = ctx.getUnsafe<Hand>(partner_agent);

        for (int cardnum = 0; cardnum < partner_hand.size; cardnum++) {
            for (int v = 0; v < bits_per_card; v++) {
                obs.bitvec[offset++] = ((partner_hand.card_plausible[cardnum] & (1 << i)) != 0);
            }

            for (int v = 0; v < colors; v++) {
                obs.bitvec[offset++] = (partner_hand.known_color[cardnum] == v);
            }
            
            for (int v = 0; v < ranks; v++) {
                obs.bitvec[offset++] = (partner_hand.known_rank[cardnum] == v);
            }
        }

        for (int cardnum = partner_hand.size; cardnum < hand_size; cardnum++) {
            for (int v = 0; v < bits_per_card + colors + ranks; v++) {
                obs.bitvec[offset++] = 0;
            }
        }
    }
    
    return offset;
}

    inline int copyObsToState(Engine &ctx, Entity &agent, int offset)
{
    Observation &obs = ctx.getUnsafe<Observation>(agent);
    State &state = ctx.getUnsafe<State>(agent);
    for (int i = 0; i < offset; i++) {
        state.bitvec[i] = obs.bitvec[i];
    }
    return offset;
}

    inline int encodeOwnHand(Engine &ctx, Entity &agent, int offset)
{
    Hand &hand = ctx.getUnsafe<Hand>(agent);
    State &obs = ctx.getUnsafe<State>(agent);

    uint32_t num_players = ctx.data().players;
    int bits_per_card = ctx.data().colors * ctx.data().ranks;
    
    for (int cardnum = 0; cardnum < hand.size; cardnum++) {
        uint8_t actual_id = hand.cards[cardnum];
        for (int b = 0; b < bits_per_card; b++) {
            obs.bitvec[offset++] = (b == actual_id);
        }
    }

    for (int cardnum = hand.size; cardnum < ctx.data().hand_size; cardnum++) {
        for (int b = 0; b < bits_per_card; b++) {
            obs.bitvec[offset++] = 0;
        }
    }

    return offset;
}

    inline void generateObsState(Engine &ctx, Entity &agent)
{
    int offset = encodeHands(ctx, agent, 0);
    offset = encodeBoard(ctx, agent, offset);
    offset = encodeDiscards(ctx, agent, offset);
    offset = encodeLastAction(ctx, agent, offset);
    offset = encodeCardKnowledge(ctx, agent, offset);

    offset = copyObsToState(ctx, agent, offset);
    offset = encodeOwnHand(ctx, agent, offset);

    (void) offset;
}

    inline void generateActionMask(Engine &ctx, Entity &agent)
{
    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;
    uint32_t num_players = ctx.data().players;
    uint32_t max_information_tokens = ctx.data().max_information_tokens;
    uint32_t max_life_tokens = ctx.data().max_life_tokens;
    uint32_t hand_size = ctx.data().hand_size;

    Deck &deck = ctx.getSingleton<Deck>();
    
    ActionMask &mask = ctx.getUnsafe<ActionMask>(agent);
    Hand &hand = ctx.getUnsafe<Hand>(agent);
    int32_t agent_id = ctx.getUnsafe<AgentID>(agent).id;
    
    int offset = 0;

    // Discard
    for (int i = 0; i < hand_size; i++) {
        mask.isValid[offset++] = (i < hand.size && deck.information_tokens < max_information_tokens);
    }

    // Play
    for (int i = 0; i < hand_size; i++) {
        mask.isValid[offset++] = (i < hand.size);
    }

    // Reveal Color
    for (int p = 1; p < num_players; p++) {
        int partner_id = (agent_id + p) % num_players;
        Entity partner_agent = ctx.data().agents[partner_id];

        Hand &partner_hand = ctx.getUnsafe<Hand>(partner_agent);
        for (int c = 0; c < colors; c++) {
            bool hascolor = false;
            for (int n = 0; n < hand_size; n++) {
                hascolor |= (partner_hand.cards[n] / ranks == c);
            }
            mask.isValid[offset++] = (deck.information_tokens > 0 && hascolor);
        }
    }

    // Reveal Rank
    for (int p = 1; p < num_players; p++) {
        int partner_id = (agent_id + p) % num_players;
        Entity partner_agent = ctx.data().agents[partner_id];

        Hand &partner_hand = ctx.getUnsafe<Hand>(partner_agent);
        for (int r = 0; r < ranks; r++) {
            bool hasrank = false;
            for (int n = 0; n < hand_size; n++) {
                hasrank |= (partner_hand.cards[n] % ranks == r);
            }
            mask.isValid[offset++] = (deck.information_tokens > 0 && hasrank);
        }
    }

    

    for (; offset < 20; offset++) {
        mask.isValid[offset] = 0;
    }

}

static void resetWorld(Engine &ctx)
{
    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    // Need to set up Deck
    Deck &deck = ctx.getSingleton<Deck>();
    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;
    uint32_t num_players = ctx.data().players;
    uint32_t max_information_tokens = ctx.data().max_information_tokens;
    uint32_t max_life_tokens = ctx.data().max_life_tokens;
    uint32_t hand_size = ctx.data().hand_size;

    int card_index = 0;
    for (int c = 0; c < colors; c++) {
        for (int r = 0; r < ranks; r++) {
            int id = ranks * c + r;
            int cr_num = (r == 0 ? 3 : r == ranks - 1 ? 1 : 2);

            for (int i = 0; i < cr_num; i++) {
                deck.cards[card_index++] = id;
            }

            deck.num_rem_cards[id] = cr_num;
            deck.discard_counts[id] = 0;
        }
    }
    deck.size = card_index;

    for (int i = 0; i < colors; i++) {
        deck.fireworks[i] = 0;
    }

    deck.information_tokens = max_information_tokens;
    deck.life_tokens = max_life_tokens;

    deck.cur_player = 0;

    deck.turns_to_play = num_players;
    deck.score = 0;
    deck.new_rew = 0;

    LastMove &lastmove = ctx.getSingleton<LastMove>();
    lastmove.move = MoveType::kInvalid;
    lastmove.player = -1;
    lastmove.target_player = -1;
    lastmove.card_index = -1;
    lastmove.scored = false;
    lastmove.information_token = false;
    lastmove.color = -1;
    lastmove.rank = -1;
    lastmove.reveal_bitmask = 0;
    lastmove.newly_revealed_bitmask = 0;
    lastmove.deal_to_player = -1;
    
    // Need to set up Observation, State, ActionMask, ActiveAgent, Hand

    uint64_t valid_mask = 1;
    valid_mask = (valid_mask << (colors * ranks)) - 1;
    for (int i = 0; i < num_players; i++) {
        Entity agent = ctx.data().agents[i];
        ctx.getUnsafe<ActiveAgent>(agent).isActive = (i == 0);
        
        for (int j = 0; j < hand_size; j++) {
            uint8_t new_card = drawDeck(ctx, deck);
            ctx.getUnsafe<Hand>(agent).cards[j] = new_card;
            ctx.getUnsafe<Hand>(agent).card_plausible[j] = valid_mask;
            ctx.getUnsafe<Hand>(agent).known_color[j] = -1;
            ctx.getUnsafe<Hand>(agent).known_rank[j] = -1;
        }
        ctx.getUnsafe<Hand>(agent).size = hand_size;

        ctx.getUnsafe<Move>(agent).type = MoveType::kInvalid;
        ctx.getUnsafe<Move>(agent).card_index = -1;
        ctx.getUnsafe<Move>(agent).target_offset = -1;
        ctx.getUnsafe<Move>(agent).color = -1;
        ctx.getUnsafe<Move>(agent).rank = -1;
    }

    for (int i = 0; i < num_players; i++) {
        Entity agent = ctx.data().agents[i];
        generateObsState(ctx, agent);
        generateActionMask(ctx, agent);
    }






    
    // for (int i = 0; i < 2; i++) {
    //     Entity agent = ctx.data().agents[i];
    
    //     ctx.getUnsafe<Location>(agent) = {
    //         static_cast<int32_t>(NUM_SPACES * ctx.data().rng.rand())
    //     };

    //     for (int t = 0; t < 2 * TIME; t++) {
    //         ctx.getUnsafe<Observation>(agent).x[t] = 0;
    //     }
    //     ctx.getUnsafe<Observation>(agent).time = ctx.getSingleton<WorldTime>().time;
    // }

    // int32_t locs[2] = {
    //     ctx.getUnsafe<Location>(ctx.data().agents[0]).x,
    //     ctx.getUnsafe<Location>(ctx.data().agents[1]).x
    // };

    // for (int i = 0; i < 2; i++) {

    //     Entity agent = ctx.data().agents[i];

    //     ctx.getUnsafe<Observation>(agent).x[0] = locs[i] + BUFFER;
    //     ctx.getUnsafe<Observation>(agent).x[TIME] = locs[1-i] + BUFFER;
    // }
}

 inline void removeFromHand(Engine &ctx, Hand &hand, int8_t index, Deck& deck)
{
    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;
    uint64_t valid_mask = 1;
    valid_mask = (valid_mask << (colors * ranks)) - 1;
    
    if (deck.size == 0) {
        // empty deck; shift everything after index in hand over by one
        for (int8_t i = index + 1; i < hand.size; i++) {
            hand.cards[i-1] = hand.cards[i];
            hand.card_plausible[i-1] = hand.card_plausible[i];
            hand.known_color[i-1] = hand.known_color[i];
            hand.known_rank[i-1] = hand.known_rank[i];
        }
        hand.size--;
        
    } else {
        // deck has cards; just insert new card into old card's location
        uint8_t newcard = drawDeck(ctx, deck);

        hand.cards[index] = newcard;
        hand.card_plausible[index] = valid_mask;
        hand.known_color[index] = -1;
        hand.known_rank[index] = -1;
        // hand doesn't change in size
    }
}

 inline void actionSystem(Engine &ctx, Deck &deck)
{
    if (deck.size == 0) {
        deck.turns_to_play--;
    }
    Entity agent = ctx.data().agents[deck.cur_player];
    LastMove &lastmove = ctx.getSingleton<LastMove>();
    
    // move is already valid by definition
    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;
    uint32_t num_players = ctx.data().players;
    uint32_t max_information_tokens = ctx.data().max_information_tokens;
    uint32_t max_life_tokens = ctx.data().max_life_tokens;
    uint32_t hand_size = ctx.data().hand_size;

    Hand &hand = ctx.getUnsafe<Hand>(agent);
    int32_t agent_id = ctx.getUnsafe<AgentID>(agent).id;
    
    int uid = ctx.getUnsafe<Action>(agent).choice;

    Move &move = ctx.getUnsafe<Move>(agent);

    uint64_t valid_mask = 1;
    valid_mask = (valid_mask << (colors * ranks)) - 1;

    // lastmove.move
    lastmove.player = deck.cur_player;
    lastmove.target_player = -1;
    lastmove.card_index = -1;
    lastmove.scored = false;
    lastmove.information_token = false;
    lastmove.color = -1;
    lastmove.rank = -1;
    lastmove.reveal_bitmask = 0;
    lastmove.newly_revealed_bitmask = 0;
    lastmove.deal_to_player = -1;

    deck.cur_player = (deck.cur_player + 1) % num_players;

    if (uid < hand_size) {
        // Discard
        move.type = MoveType::kDiscard;
        move.card_index = uid;
        move.target_offset = -1;
        move.color = -1;
        move.rank = -1;

        lastmove.move = move.type;
        lastmove.card_index = uid;
        uint8_t cardval = hand.cards[uid];
        lastmove.color = cardval / ranks;
        lastmove.rank = cardval % ranks;

        deck.discard_counts[cardval]++;
        deck.information_tokens++;

        // draw from deck to replace card in hand
        removeFromHand(ctx, hand, uid, deck);
        return;
    }

    uid -= hand_size;

    if (uid < hand_size) {
        // Play
        move.type = MoveType::kPlay;
        move.card_index = uid;
        move.target_offset = -1;
        move.color = -1;
        move.rank = -1;

        lastmove.move = move.type;
        lastmove.card_index = uid;
        uint8_t cardval = hand.cards[uid];
        lastmove.color = cardval / ranks;
        lastmove.rank = cardval % ranks;

        // attempt to add to fireworks
        if (deck.fireworks[lastmove.color] == lastmove.rank) {
            deck.fireworks[lastmove.color]++;
            if (deck.fireworks[lastmove.color] == ranks) {
                deck.information_tokens++;
                lastmove.information_token = true;
            } else {
                lastmove.information_token = false;
            }
            lastmove.scored = true;
        } else {
            deck.discard_counts[cardval]++;
            deck.life_tokens--;
            lastmove.scored = false;
        }
        
        // draw from deck to replace card in hand
        removeFromHand(ctx, hand, uid, deck);
        return;
    }

    uid -= hand_size;

    if (uid < (num_players - 1) * colors) {
        // reveal color
        move.type = MoveType::kRevealColor;
        move.card_index = -1;
        move.target_offset = 1 + uid / colors;
        move.color = uid % colors;
        move.rank = -1;

        deck.information_tokens--;

        int partner_id = (agent_id + move.target_offset) % num_players;
        Entity partner_agent = ctx.data().agents[partner_id];

        Hand &partner_hand = ctx.getUnsafe<Hand>(partner_agent);

        lastmove.target_player = partner_id;
        lastmove.color = move.color;

        lastmove.reveal_bitmask = 0;
        for (int i = 0; i < partner_hand.size; i++) {
            if (partner_hand.cards[i] / ranks == lastmove.color) {
                lastmove.reveal_bitmask |= static_cast<uint8_t>(1) << i;
            }
        }

        lastmove.newly_revealed_bitmask = 0;
        uint64_t newmask = 0;
        for (int i = 0; i < ranks; i++) {
            newmask |= static_cast<uint8_t>(1) << (lastmove.color * ranks + i);
        }
        for (int i = 0; i < partner_hand.size; i++) {
            if (partner_hand.cards[i] / ranks == lastmove.color) {
                if (partner_hand.known_color[i] == -1) {
                    lastmove.newly_revealed_bitmask |= static_cast<uint8_t>(1) << i;
                }

                // apply is color hint
                partner_hand.known_color[i] = lastmove.color;
                partner_hand.card_plausible[i] &= newmask;
            } else {
                // apply not color hint
                partner_hand.card_plausible[i] &= ~newmask;
            }
        }
        return;
    }

    uid -= (num_players - 1) * colors;

    {
        // reveal rank
        move.type = MoveType::kRevealRank;
        move.card_index = -1;
        move.target_offset = 1 + uid / ranks;
        move.color = -1;
        move.rank = uid % ranks;

        deck.information_tokens--;

        int partner_id = (agent_id + move.target_offset) % num_players;
        Entity partner_agent = ctx.data().agents[partner_id];

        Hand &partner_hand = ctx.getUnsafe<Hand>(partner_agent);

        lastmove.target_player = partner_id;
        lastmove.rank = move.rank;

        lastmove.reveal_bitmask = 0;
        for (int i = 0; i < partner_hand.size; i++) {
            if (partner_hand.cards[i] % ranks == lastmove.rank) {
                lastmove.reveal_bitmask |= static_cast<uint8_t>(1) << i;
            }
        }

        lastmove.newly_revealed_bitmask = 0;
        uint64_t newmask = 0;
        for (int i = 0; i < ranks; i++) {
            newmask |= static_cast<uint8_t>(1) << (i * ranks + lastmove.rank);
        }
        for (int i = 0; i < partner_hand.size; i++) {
            if (partner_hand.cards[i] % ranks == lastmove.rank) {
                if (partner_hand.known_color[i] == -1) {
                    lastmove.newly_revealed_bitmask |= static_cast<uint8_t>(1) << i;
                }

                // apply is color hint
                partner_hand.known_rank[i] = lastmove.rank;
                partner_hand.card_plausible[i] &= newmask;
            } else {
                // apply not color hint
                partner_hand.card_plausible[i] &= ~newmask;
            }
        }
    }

    

}

    inline void observationSystem(Engine &ctx, Deck &deck)
{
    uint32_t num_players = ctx.data().players;

    // only update for cur_player
    // Entity agent = ctx.data().agents[deck.cur_player];
    for (int i = 0; i < num_players; i++) {
        Entity agent = ctx.data().agents[i];

        if (i == deck.cur_player) {
            ctx.getUnsafe<ActiveAgent>(agent).isActive = 1;
            generateObsState(ctx, agent);
            generateActionMask(ctx, agent);
        } else {
            ctx.getUnsafe<ActiveAgent>(agent).isActive = 0;
        }
    }
}
    
    inline void checkDone(Engine &ctx, WorldReset &reset)
{
    reset.resetNow = false;
    Deck &deck = ctx.getSingleton<Deck>();
    uint32_t colors = ctx.data().colors;
    uint32_t ranks = ctx.data().ranks;
    uint32_t num_players = ctx.data().players;
    uint32_t max_information_tokens = ctx.data().max_information_tokens;
    uint32_t max_life_tokens = ctx.data().max_life_tokens;
    uint32_t hand_size = ctx.data().hand_size;

    int8_t old_score = deck.score;
    deck.score = 0;
    if (deck.life_tokens > 0) {
        for (int i = 0; i < colors; i++) {
            deck.score += deck.fireworks[i];
        }
    }
    deck.new_rew = deck.score - old_score;

    for (int i = 0; i < num_players; i++) {
        Entity agent = ctx.data().agents[i];
        ctx.getUnsafe<Reward>(agent).rew = deck.new_rew;
    }

    if (deck.life_tokens < 1) {
        reset.resetNow = true;
    }

    if (deck.score >= colors * ranks) {
        reset.resetNow = true;
    }

    if (deck.turns_to_play <= 0) {
        reset.resetNow = true;
    }

    if (reset.resetNow) {
        resetWorld(ctx);
    }
}
    

void Sim::setupTasks(TaskGraph::Builder &builder, const Config &)
{   
    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
                                                     Deck>>({});

    auto update_obs = builder.addToGraph<ParallelForNode<Engine, observationSystem,
                                                         Deck>>({action_sys});

    auto terminate_sys = builder.addToGraph<ParallelForNode<Engine, checkDone, WorldReset>>({update_obs});

    (void)terminate_sys;
    // (void) action_sys;

    // printf("Setup done\n");
}


Sim::Sim(Engine &ctx, const Config& cfg,  const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      colors(init.colors),
      ranks(init.ranks),
      players(init.players),
      max_information_tokens(init.max_information_tokens),
      max_life_tokens(init.max_life_tokens)
{
    hand_size = (players < 4 ? 5 : 4);
    // Make a buffer that will last the duration of simulation for storing
    // agent entity IDs
    agents = (Entity *)rawAlloc(players * sizeof(Entity));

    for (int i = 0; i < players; i++) {
        agents[i] = ctx.makeEntityNow<Agent>();
        ctx.getUnsafe<Action>(agents[i]).choice = 0;
        ctx.getUnsafe<AgentID>(agents[i]).id = i;
        ctx.getUnsafe<Reward>(agents[i]).rew = 0.f;
        ctx.getUnsafe<ActiveAgent>(agents[i]).isActive = true;
    }
    // Initial reset
    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;

    // printf("Did initial reset\n");
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
