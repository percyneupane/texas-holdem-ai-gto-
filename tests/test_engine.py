import random
import unittest
from dataclasses import replace

from poker_research.cards import Card
from poker_research.engine import (BIG_BLIND, SMALL_BLIND, STARTING_STACK, Seat, apply_action, current_hand_label,
                                   describe_hand, legal_actions, side_pots, start_hand)
from poker_research.opponents import hand_strength, opponent_action


def table(count, stacks=None):
    stacks = stacks or [STARTING_STACK] * count
    return tuple(Seat("P%d" % i, stack=stacks[i]) for i in range(count))


def rig(state, hands, board):
    """Give each seat a fixed hand and stack the deck so the board comes out as given."""
    seats = tuple(replace(seat, hand=tuple(hands[i])) if i < len(hands) else seat
                  for i, seat in enumerate(state.seats))
    return replace(state, seats=seats, deck=tuple(board))


class EngineTests(unittest.TestCase):
    def test_blinds_posted_left_of_dealer_and_utg_acts_first(self):
        state = start_hand(table(6), dealer=0, rng=random.Random(1))
        self.assertEqual(state.seats[1].bet, SMALL_BLIND)
        self.assertEqual(state.seats[2].bet, BIG_BLIND)
        self.assertEqual(state.to_act, 3)
        self.assertEqual(state.pot, SMALL_BLIND + BIG_BLIND)

    def test_heads_up_dealer_posts_small_blind_and_acts_first(self):
        state = start_hand(table(2), dealer=1, rng=random.Random(1))
        self.assertEqual(state.seats[1].bet, SMALL_BLIND)
        self.assertEqual(state.seats[0].bet, BIG_BLIND)
        self.assertEqual(state.to_act, 1)

    def test_everyone_folds_to_big_blind_who_wins_the_blinds(self):
        state = start_hand(table(4), dealer=0, rng=random.Random(2))
        while state.to_act is not None:
            state = apply_action(state, "fold")
        self.assertTrue(state.complete)
        self.assertFalse(state.showdown)
        self.assertEqual(state.seats[2].stack, STARTING_STACK + SMALL_BLIND)
        self.assertEqual(state.awards[0].seat, 2)

    def test_big_blind_gets_option_after_limps(self):
        state = start_hand(table(3), dealer=0, rng=random.Random(3))
        state = apply_action(state, "call")  # dealer limps
        state = apply_action(state, "call")  # small blind completes
        self.assertEqual(state.to_act, 2)
        self.assertEqual(legal_actions(state).to_call, 0)
        state = apply_action(state, "check")
        self.assertEqual(state.street, "flop")
        self.assertEqual(len(state.board), 3)
        self.assertEqual(state.to_act, 1)  # first live seat left of the dealer

    def test_raise_sets_new_minimum_raise(self):
        state = start_hand(table(4), dealer=0, rng=random.Random(4))
        state = apply_action(state, "raise", 40)
        self.assertEqual(state.current_bet, 40)
        self.assertEqual(state.min_raise, 30)
        self.assertEqual(legal_actions(state).min_raise_to, 70)

    def test_raise_below_minimum_is_lifted_to_the_minimum(self):
        state = start_hand(table(4), dealer=0, rng=random.Random(4))
        state = apply_action(state, "raise", 11)
        self.assertEqual(state.current_bet, 2 * BIG_BLIND)

    def test_invalid_check_becomes_fold_and_unknown_action_rejected(self):
        state = start_hand(table(3), dealer=0, rng=random.Random(5))
        after = apply_action(state, "check")
        self.assertTrue(after.seats[0].folded)
        with self.assertRaises(ValueError):
            apply_action(state, "dance")

    def test_all_in_runs_out_board_and_pays_side_pot(self):
        state = start_hand(table(3, [50, 1000, 1000]), dealer=0, rng=random.Random(6))
        hands = [[Card(14, "s"), Card(14, "h")], [Card(13, "s"), Card(13, "h")], [Card(2, "c"), Card(7, "d")]]
        board = [Card(3, "c"), Card(8, "d"), Card(9, "h"), Card(12, "s"), Card(4, "d")]
        state = rig(state, hands, board)
        state = apply_action(state, "allin")          # P0 all-in for 50
        state = apply_action(state, "raise", 200)     # P1 raises
        state = apply_action(state, "call")           # P2 calls 200
        while state.to_act is not None:
            state = apply_action(state, "check")
        self.assertTrue(state.showdown)
        awards = {award.seat: award.amount for award in state.awards}
        self.assertEqual(awards, {0: 150, 1: 300})   # aces win the main pot, kings the side pot
        self.assertEqual(sum(seat.stack for seat in state.seats), 2050)

    def test_split_pot_shares_chips(self):
        state = start_hand(table(2), dealer=0, rng=random.Random(7))
        hands = [[Card(2, "c"), Card(3, "d")], [Card(2, "h"), Card(3, "s")]]
        board = [Card(10, "c"), Card(11, "d"), Card(12, "h"), Card(13, "s"), Card(14, "d")]
        state = apply_action(rig(state, hands, board), "call")
        while state.to_act is not None:
            state = apply_action(state, "check")
        self.assertEqual([award.amount for award in state.awards], [10, 10])
        self.assertTrue(all(award.hand_name == "Straight, Ace high" for award in state.awards))

    def test_side_pots_merge_folded_overcommit(self):
        seats = (Seat("A", hand=(Card(2, "c"), Card(3, "c")), committed=50),
                 Seat("B", hand=(Card(4, "c"), Card(5, "c")), committed=100, folded=True),
                 Seat("C", hand=(Card(6, "c"), Card(7, "c")), committed=80))
        self.assertEqual(side_pots(seats), [(150, (0, 2)), (80, (2,))])

    def test_random_hands_conserve_chips_and_always_finish(self):
        rng = random.Random(42)
        seats = table(6)
        for dealer in range(300):
            state = start_hand(seats, dealer % 6, rng)
            while state.to_act is not None:
                state = apply_action(state, *opponent_action(state, rng))
            self.assertTrue(state.complete)
            self.assertEqual(sum(seat.stack for seat in state.seats), 6 * STARTING_STACK)
            seats = tuple(replace(seat, stack=STARTING_STACK) if seat.stack == 0 else seat for seat in state.seats)
            seats = table(6) if sum(s.stack for s in seats) != 6 * STARTING_STACK else seats

    def test_start_hand_requires_two_players(self):
        with self.assertRaises(ValueError):
            start_hand(table(2, [1000, 0]), 0, random.Random(1))

    def test_hand_descriptions(self):
        self.assertEqual(describe_hand((1, (6, 14, 9, 2))), "Pair of Sixes")
        self.assertEqual(describe_hand((8, (14,))), "Royal flush")
        self.assertEqual(current_hand_label([Card(14, "s"), Card(9, "h")], []), "High card, Ace")
        self.assertEqual(current_hand_label([Card(9, "s"), Card(9, "h")], [Card(9, "c"), Card(2, "d"), Card(5, "s")]),
                         "Three of a kind, Nines")

    def test_hand_strength_ranks_premium_hands_higher(self):
        aces = hand_strength([Card(14, "s"), Card(14, "h")], [])
        junk = hand_strength([Card(7, "s"), Card(2, "h")], [])
        self.assertGreater(aces, junk)


if __name__ == "__main__":
    unittest.main()
