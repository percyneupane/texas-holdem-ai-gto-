import unittest

from poker_research.turn_detection import action_controls


def label(text, x=0.25, y=0.65):
    return dict(text=text, confidence=0.99, x=x, y=y, width=0.1, height=0.025)


class ActionTextTests(unittest.TestCase):
    def test_controls_can_be_above_the_old_strip(self):
        rows = [label('Fold', y=0.6), label('Call $0.05', x=0.5, y=0.6),
                label('Raise to 0.15', x=0.75, y=0.6)]
        self.assertEqual([name for name, _ in action_controls(rows)], ['FOLD', 'CALL', 'RAISE'])

    def test_seat_action_badges_do_not_make_a_button_row(self):
        self.assertFalse(action_controls([label('FOLD', y=0.5), label('CALL', y=0.7)]))
        self.assertFalse(action_controls([label('FOLD', y=0.2), label('CHECK', y=0.2)]))

    def test_preselection_and_sitting_out_do_not_start_a_read(self):
        for rows in ([label('Check/Fold'), label('Call any')],
                     [label('I AM BACK')], [label('REJOIN HAND')], []):
            with self.subTest(rows=rows):
                self.assertFalse(action_controls(rows))

    def test_check_and_fold_are_a_valid_free_action(self):
        self.assertTrue(action_controls([label('FOLD'), label('CHECK', x=0.5)]))


if __name__ == '__main__':
    unittest.main()
