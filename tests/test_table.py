import json
import random
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

from poker_research.engine import Seat, apply_action, legal_actions, start_hand
from poker_research.table_display import (
    Handler, local_table_url, parse_act, parse_click, submit_observation, submit_table_action, whole_chips,
)
from poker_research.table_game import HERO, TableGame, resize_seats
from poker_research.table_layout import BUTTONS, SEAT_LAYOUTS, box_center, hit_test
from poker_research.table_render import render_table_image
from poker_research.table_view import build_view
from poker_research.vision import NotHeroTurn, TableNotFoundError, decode_table_image


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def hero_turn_state(players, seed):
    """Advance a fresh hand (folding everyone in front) until the hero is due to act."""
    state = start_hand(tuple(Seat("P%d" % i) for i in range(players)), 1 % players, random.Random(seed))
    while state.to_act != HERO:
        state = apply_action(state, "call")
    return state


def play_until(game, clock, condition, limit=500):
    for _ in range(limit):
        if condition():
            return True
        clock.now += 1.0
        game.tick()
    return condition()


class RenderAndReadTests(unittest.TestCase):
    def test_reader_round_trips_live_tables_of_every_size(self):
        for players in range(2, 11):
            state = hero_turn_state(players, seed=players)
            view = build_view(state, HERO)
            decoded = decode_table_image(render_table_image(view))
            self.assertEqual(decoded["hero"], view["hero"], players)
            self.assertEqual(decoded["board"], view["board"], players)
            self.assertEqual(decoded["pot"], view["pot"], players)
            self.assertEqual(decoded["to_call"], view["to_call"], players)
            self.assertEqual(decoded["stack"], view["stack"], players)
            self.assertEqual(decoded["active_players"], view["active_players"], players)
            self.assertEqual(decoded["min_raise"], view["raise_to"] - view["hero_bet"] - view["to_call"], players)

    def test_reader_uses_center_board_bottom_center_hero_and_surrounding_backs(self):
        state = hero_turn_state(7, seed=17)
        view = build_view(state, HERO)
        decoded = decode_table_image(render_table_image(view))

        self.assertEqual(decoded["board"], view["board"])
        self.assertEqual(decoded["hero"], view["hero"])
        self.assertEqual(decoded["active_players"], 7)

    def test_reader_waits_when_it_is_not_the_heros_turn(self):
        state = start_hand(tuple(Seat("P%d" % i) for i in range(6)), 0, random.Random(3))
        self.assertNotEqual(state.to_act, HERO)
        with self.assertRaises(NotHeroTurn):
            decode_table_image(render_table_image(build_view(state, HERO)))

    def test_reader_rejects_frames_without_the_table(self):
        from PIL import Image
        with self.assertRaises(TableNotFoundError):
            decode_table_image(Image.new("RGB", (1000, 720), (200, 200, 200)))

    def test_showdown_reveals_opponent_cards_and_winner(self):
        state = start_hand(tuple(Seat("P%d" % i) for i in range(3)), 0, random.Random(9))
        while state.to_act is not None:
            state = apply_action(state, "call")
        view = build_view(state, HERO)
        self.assertTrue(state.showdown)
        self.assertTrue(all(seat["cards"] in ("faces", "hero") for seat in view["seats"]))
        self.assertTrue(any(seat["kind"] == "win" for seat in view["seats"]))
        self.assertIn("wins", view["message"])
        render_table_image(view)  # draws without error

    def test_short_handed_tables_use_balanced_slots(self):
        for count, slots in SEAT_LAYOUTS.items():
            self.assertEqual(len(slots), count)
            self.assertEqual(slots[0], 0)


class TableGameTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.game = TableGame(players=4, seed=5, clock=self.clock)

    def test_first_tick_deals_a_hand(self):
        self.game.tick()
        self.assertEqual(self.game.hand_number, 1)
        self.assertEqual(len(self.game.view()["seats"]), 4)

    def test_opponents_act_until_hero_turn_and_hero_can_fold(self):
        self.assertTrue(play_until(self.game, self.clock,
                                   lambda: self.game.state is not None and self.game.state.to_act == HERO))
        self.game.click("fold")
        self.assertTrue(self.game.state.seats[HERO].folded or self.game.state.complete)

    def test_sizing_buttons_clamp_raise_amount(self):
        play_until(self.game, self.clock, lambda: self.game.state is not None and self.game.state.to_act == HERO)
        self.game.click("allin")
        self.assertEqual(self.game.raise_to, self.game.state.seats[HERO].bet + self.game.state.seats[HERO].stack)
        self.game.click("slider", x=0)
        self.assertEqual(self.game.view()["raise_to"], self.game.view()["min_raise_to"])

    def test_timebank_expiry_checks_or_folds_for_the_hero(self):
        play_until(self.game, self.clock, lambda: self.game.state is not None and self.game.state.to_act == HERO)
        version = self.game.version
        self.clock.now += 60
        self.game.tick()
        self.assertGreater(self.game.version, version)
        self.assertNotEqual(self.game.state.to_act, HERO)

    def test_autoplay_plays_many_hands_and_player_count_changes_next_hand(self):
        self.game.click("autoplay")
        self.game.click("players_plus")
        self.game.click("players_plus")
        self.assertTrue(play_until(self.game, self.clock, lambda: self.game.hand_number >= 6, limit=3000))
        self.assertEqual(len(self.game.state.seats), 6)
        acted = [spot for hand in self.game.session.as_dict()['table_hands']
                 for spot in hand['hero_actions'] if spot.get('equity') is not None]
        self.assertTrue(acted)
        self.assertTrue(all(spot.get('reason') for spot in acted))
        key, png = self.game.frame_png()
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertEqual(self.game.frame_png()[0], key)
        self.assertTrue(any("Hand #" in line for line in self.game.log()))

    def test_autoplay_from_start_records_decisions_and_can_be_paused(self):
        game = TableGame(players=2, seed=5, clock=self.clock,
                         autoplay=True, policy='baseline')
        self.addCleanup(game.close)
        self.assertEqual(game.agent.policy_name, 'baseline')
        self.assertTrue(play_until(game, self.clock, lambda: game.hand_number >= 4,
                                   limit=1500))
        hands = game.session.as_dict()['table_hands']
        self.assertEqual(len(hands), 3)
        actions = [spot for hand in hands for spot in hand['hero_actions']]
        self.assertTrue(actions)
        self.assertTrue(all(spot['equity'] is not None for spot in actions))
        game.click('autoplay')
        self.assertFalse(game.autoplay)
        self.assertTrue(play_until(game, self.clock, lambda: game.state.to_act == HERO))
        state = game.state
        self.clock.now += 2
        game.tick()
        self.assertIs(game.state, state)

    def test_manual_start_is_default_and_unknown_policy_is_rejected(self):
        self.assertFalse(self.game.autoplay)
        self.assertEqual(self.game.agent.policy_name, 'contextual')
        with self.assertRaisesRegex(ValueError, 'unknown policy'):
            TableGame(policy='missing')

    def test_player_count_is_bounded(self):
        for _ in range(20):
            self.game.click("players_minus")
        self.assertEqual(self.game.player_count, 2)
        with self.assertRaises(ValueError):
            TableGame(players=11)

    def test_resize_seats_rebuys_busted_players(self):
        seats = (Seat("You", stack=0), Seat("Ana", stack=300), Seat("Bo"))
        resized = resize_seats(seats, 5)
        self.assertEqual(len(resized), 5)
        self.assertEqual(resized[0].stack, 1000)
        self.assertEqual(resized[1].stack, 300)


class ClickTests(unittest.TestCase):
    def test_hit_test_maps_buttons(self):
        for name, box in BUTTONS.items():
            self.assertEqual(hit_test(*box_center(box)), name)
        self.assertIsNone(hit_test(500, 300))

    def test_parse_click_validates_payload(self):
        self.assertEqual(parse_click(json.dumps({"x": 10, "y": 20}).encode()), (10, 20))
        for bad in ({"x": "1", "y": 2}, {"x": 5000, "y": 2}, {"x": True, "y": 2}):
            with self.assertRaises(ValueError):
                parse_click(json.dumps(bad).encode())


class ActTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.game = TableGame(players=4, seed=5, clock=self.clock)

    def _hero_turn(self):
        self.assertTrue(play_until(
            self.game, self.clock,
            lambda: self.game.state is not None and self.game.state.to_act == HERO))

    def test_act_raises_by_chips_added_and_refuses_a_second_action(self):
        self._hero_turn()
        legal = legal_actions(self.game.state)
        chips = legal.min_raise_to - self.game.state.seats[HERO].bet
        self.assertGreater(chips, 0)
        log = self.game.act("raise", chips)
        self.assertIn("Hero", log)
        self.assertIn(str(legal.min_raise_to), log)
        self.assertNotEqual(self.game.state.to_act, HERO)
        version = self.game.version
        with self.assertRaises(ValueError):
            self.game.act("fold")
        self.assertEqual(self.game.version, version)

    def test_act_refuses_a_sizeless_raise_and_a_turn_that_has_not_started(self):
        self._hero_turn()
        version = self.game.version
        with self.assertRaises(ValueError):
            self.game.act("bet", 0)
        self.assertEqual(self.game.version, version)
        self.assertEqual(self.game.state.to_act, HERO)
        idle = TableGame(players=4, seed=5, clock=FakeClock())
        with self.assertRaises(ValueError):
            idle.act("fold")

    def test_amounts_and_urls_stay_on_whole_local_chips(self):
        self.assertEqual(whole_chips(25), 25)
        self.assertEqual(whole_chips(25.0), 25)
        self.assertEqual(parse_act(json.dumps({"action": "Fold"}).encode()), ("fold", 0, {}))
        self.assertEqual(parse_act(json.dumps({"action": "raise", "amount": 40}).encode()), ("raise", 40, {}))
        noted = parse_act(json.dumps({
            "action": "call", "equity": 0.25, "reason": "opening chart", "bluff_probability": 0,
        }).encode())
        self.assertEqual(noted[0], "call")
        self.assertEqual(noted[2]["equity"], 0.25)
        self.assertEqual(noted[2]["reason"], "opening chart")
        for bad in (True, -1, 25.5, float("nan"), "25"):
            with self.assertRaises(ValueError):
                whole_chips(bad)
        for bad in ({"action": "allin"}, {"action": 1}, ["fold"], {"action": "bet", "amount": 1.5},
                    {"action": "fold", "equity": 2}, {"action": "fold", "reason": 1}):
            with self.assertRaises(ValueError):
                parse_act(json.dumps(bad).encode())
        self.assertEqual(local_table_url("http://127.0.0.1:8780/"), "http://127.0.0.1:8780")
        for bad in ("https://127.0.0.1:8780", "http://example.com", "http://user:pass@127.0.0.1:8780"):
            with self.assertRaises(ValueError):
                local_table_url(bad)
        with self.assertRaises(ValueError):
            submit_table_action("http://127.0.0.1:9", "raise", 1.25)

    def test_submit_plays_one_fold_on_the_local_table(self):
        self._hero_turn()
        previous = Handler.game
        server = None
        try:
            Handler.game = self.game
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = "http://127.0.0.1:%d" % server.server_address[1]
            log = None
            last = None
            deadline = time.time() + 2
            while time.time() < deadline:
                try:
                    log = submit_table_action(url, "fold", 0, timeout=0.5)
                    break
                except ConnectionError as error:
                    last = error
                    time.sleep(0.05)
            self.assertIsNotNone(log, last)
            self.assertIn("Hero", log)
            self.assertTrue(self.game.state.complete or self.game.state.to_act != HERO)
            with self.assertRaisesRegex(ValueError, "Not your turn"):
                submit_table_action(url, "call", 0, timeout=0.5)
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            Handler.game = previous

    def test_a_finished_hand_keeps_the_result_and_the_reason(self):
        self._hero_turn()
        self.game.act('fold', 0, note={'equity': 0.33, 'reason': 'chart fold', 'bluff_probability': 0})
        self.assertTrue(play_until(
            self.game, self.clock, lambda: self.game.session.totals()['hands'] >= 1, limit=800))
        data = self.game.session_snapshot()
        self.assertEqual(data['table']['profit_chips'],
                         sum(hand['profit_chips'] for hand in data['table_hands']))
        hand = data['table_hands'][0]
        self.assertEqual(hand['profit_chips'], hand['ending_stack'] - hand['starting_stack'])
        self.assertEqual(len(hand['hero']), 2)
        self.assertTrue(hand['log'])
        self.assertTrue(hand['actions'])
        reasons = [spot.get('reason') for item in data['table_hands'] for spot in item['hero_actions']]
        self.assertIn('chart fold', reasons)
        self.assertIn('not a real-money profit edge', data['note'])
        self.assertIn('Open the session page.', self.game.session_line())

    def test_session_endpoint_records_an_overlay_read_apart_from_chip_profit(self):
        self._hero_turn()
        self.game.act('fold', 0, note={'equity': 0.1, 'reason': 'chart fold'})
        self.assertTrue(play_until(
            self.game, self.clock, lambda: self.game.session.totals()['hands'] >= 1, limit=800))
        profit = self.game.session.totals()['profit_chips']
        previous = Handler.game
        server = None
        try:
            Handler.game = self.game
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = "http://127.0.0.1:%d" % server.server_address[1]
            with urlopen(url + '/session.json', timeout=1) as response:
                body = json.loads(response.read().decode('utf-8'))
            self.assertGreaterEqual(body['table']['hands'], 1)
            self.assertEqual(body['table']['profit_chips'], profit)
            with urlopen(url + '/session', timeout=1) as response:
                page = response.read().decode('utf-8')
                self.assertIn('text/html', response.headers.get('Content-Type', ''))
            self.assertIn('id="hands"', page)
            self.assertIn("fetch('/session.json'", page)
            with urlopen(url + '/session?hand=1', timeout=1) as response:
                self.assertEqual(response.status, 200)
            with urlopen(url + '/', timeout=1) as response:
                table = response.read().decode('utf-8')
            self.assertIn('href="/session"', table)
            self.assertNotIn('id="session"></div>', table)
            submit_observation(url, {
                'hero': [[14, 's'], [13, 'h']], 'board': [], 'action': 'raise', 'amount': 25,
                'equity': 0.55, 'reason': 'opening chart', 'position': 'BTN', 'pot': 15,
                'to_call': 10, 'stack': 990, 'big_blind': 10,
            })
            submit_observation(url, {
                'hero': ['As', 'Kh'], 'board': [], 'action': 'raise', 'amount': 25,
                'equity': 0.55, 'reason': 'opening chart', 'pot': 15,
            })
            with urlopen(url + '/session.json', timeout=1) as response:
                body = json.loads(response.read().decode('utf-8'))
            self.assertEqual(body['observed']['decisions'], 1)
            self.assertEqual(body['observed_hands'][0]['hero'], ['As', 'Kh'])
            self.assertEqual(body['table']['profit_chips'], profit)
            with urlopen(url + '/state.json', timeout=1) as response:
                state = json.loads(response.read().decode('utf-8'))
            self.assertIn('Research table', state['session'])
            with self.assertRaises(ValueError):
                submit_observation(url, {'hero': ['As'], 'action': 'fold'})
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            Handler.game = previous


if __name__ == "__main__":
    unittest.main()
