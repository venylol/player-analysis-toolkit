from __future__ import annotations

import pandas as pd
import unittest

from src.labels import assert_no_label_leakage, decision_nodes, disc_loss_to_severity_class, generate_disc_loss_labels


class LabelTests(unittest.TestCase):
  def test_adjacent_formulas_and_pass_rows_are_not_decisions(self):
    frame = pd.DataFrame([
        {"game_id": "g", "move_index": 0, "ply": 1, "side_to_move": "black", "actual_move": "d3", "hint6_1_score": 5},
        {"game_id": "g", "move_index": 1, "ply": 2, "side_to_move": "white", "actual_move": "-", "hint6_1_score": None},
        {"game_id": "g", "move_index": 2, "ply": 3, "side_to_move": "black", "actual_move": "c4", "hint6_1_score": 7},
        {"game_id": "g", "move_index": 3, "ply": 4, "side_to_move": "white", "actual_move": "f5", "hint6_1_score": 2},
    ])
    labelled = generate_disc_loss_labels(frame)
    self.assertEqual(labelled.loc[0, "raw_loss"], -2)
    self.assertEqual(labelled.loc[0, "disc_loss"], 0)
    self.assertEqual(labelled.loc[0, "child_pass_count"], 1)
    self.assertTrue(labelled.loc[0, "same_side_after_move"])
    self.assertEqual(labelled.loc[2, "raw_loss"], 9)
    self.assertFalse(labelled.loc[1, "label_available"])
    nodes = decision_nodes(labelled)
    self.assertEqual(nodes["actual_move"].tolist(), ["d3", "c4", "f5"])
    self.assertEqual(nodes["global_placement_ply"].tolist(), [1, 2, 3])
    self.assertEqual(nodes["ply"].tolist(), [1, 2, 3])

  def test_broken_source_continuity_does_not_create_a_label(self):
    frame = pd.DataFrame([
        {"game_id": "g", "move_index": 0, "ply": 1, "side_to_move": "black", "actual_move": "d3", "hint6_1_score": 5},
        {"game_id": "g", "move_index": 2, "ply": 3, "side_to_move": "white", "actual_move": "c4", "hint6_1_score": 4},
    ])
    labelled = generate_disc_loss_labels(frame)
    self.assertFalse(labelled.loc[0, "has_consecutive_child"])
    self.assertFalse(labelled.loc[0, "label_available"])


  def test_negative_raw_loss_is_clipped_but_retained(self):
    frame = pd.DataFrame([
        {"game_id": "g", "move_index": 0, "ply": 1, "side_to_move": "black", "actual_move": "d3", "hint6_1_score": 1},
        {"game_id": "g", "move_index": 1, "ply": 2, "side_to_move": "black", "actual_move": "c3", "hint6_1_score": 4},
    ])
    labelled = generate_disc_loss_labels(frame)
    self.assertEqual(labelled.loc[0, "raw_loss"], -3)
    self.assertEqual(labelled.loc[0, "disc_loss"], 0)


  def test_model_input_leakage_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "cannot enter model input"):
        assert_no_label_leakage(["ply", "next_best_score"])

  def test_all_loss_history_forms_are_omitted_not_masked(self):
    forbidden = [
        "own_previous_disc_loss", "own_raw_loss_history", "own_zero_loss_rate",
        "own_loss_ge4_count", "own_loss_ge10_rate", "own_cumulative_loss",
        "own_average_loss", "own_recent_loss", "own_predicted_loss_probability",
        "own_previous_residual", "own_disc_loss__missing",
        "own_previous_probability_ge4", "ownPreviousProbabilityGe10",
        "own_previous_zero_rate",
    ]
    for name in forbidden:
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, "must be omitted"):
        assert_no_label_leakage(["ply", name])

  def test_source_ply_72_with_twelve_passes_becomes_sixty_placements(self):
    rows = []
    pass_indexes = set(range(5, 72, 6))
    for index in range(72):
      rows.append({
          "game_id": "long", "move_index": index, "ply": index + 1,
          "side_to_move": "black" if index % 2 == 0 else "white",
          "actual_move": "-" if index in pass_indexes else "d3",
          "hint6_1_score": 0,
      })
    labelled = generate_disc_loss_labels(pd.DataFrame(rows))
    nodes = decision_nodes(labelled)
    self.assertEqual(int(labelled["source_ply_including_pass"].max()), 72)
    self.assertEqual(len(nodes), 60)
    self.assertEqual(int(nodes["global_placement_ply"].max()), 60)

  def test_four_class_boundaries_are_exact(self):
    losses = pd.Series([0, 1, 3, 4, 9, 10])
    self.assertEqual(disc_loss_to_severity_class(losses).astype(int).tolist(), [0, 1, 1, 2, 2, 3])

  def _wld_transition(self, before, next_score):
    rows = [
        {"game_id": "w", "move_index": i, "ply": i + 1,
         "side_to_move": "black" if i % 2 == 0 else "white",
         "actual_move": "d3", "hint6_1_score": 0}
        for i in range(40)
    ]
    rows[38]["hint6_1_score"] = before
    rows[39]["hint6_1_score"] = next_score
    return generate_disc_loss_labels(pd.DataFrame(rows))

  def test_wld_ply_boundary_and_rank_drops(self):
    cases = ((1, 0, 1, 0.5), (0, 1, 1, 0.5), (1, 1, 2, 1.0), (-1, -1, 0, 0.0))
    for before, next_score, expected_class, expected_loss in cases:
      with self.subTest(before=before, next=next_score):
        labelled = self._wld_transition(before, next_score)
        self.assertFalse(bool(labelled.loc[37, "wld_label_available"]))
        self.assertTrue(bool(labelled.loc[38, "wld_label_available"]))
        self.assertEqual(int(labelled.loc[38, "wld_class"]), expected_class)
        self.assertEqual(float(labelled.loc[38, "wld_loss"]), expected_loss)

  def test_wld_normal_turn_flips_and_pass_same_side_does_not(self):
    normal = self._wld_transition(2, 3)
    self.assertEqual(float(normal.loc[38, "actual_move_score"]), -3.0)
    rows = [
        {"game_id": "p", "move_index": i, "ply": i + 1,
         "side_to_move": "black" if i % 2 == 0 else "white",
         "actual_move": "d3", "hint6_1_score": 0}
        for i in range(39)
    ]
    rows[38]["side_to_move"] = "black"
    rows[38]["hint6_1_score"] = 2
    rows.extend([
        {"game_id": "p", "move_index": 39, "ply": 40, "side_to_move": "white", "actual_move": "-", "hint6_1_score": None},
        {"game_id": "p", "move_index": 40, "ply": 41, "side_to_move": "black", "actual_move": "c4", "hint6_1_score": -3},
    ])
    passed = generate_disc_loss_labels(pd.DataFrame(rows))
    self.assertTrue(bool(passed.loc[38, "same_side_after_move"]))
    self.assertEqual(float(passed.loc[38, "actual_move_score"]), -3.0)
