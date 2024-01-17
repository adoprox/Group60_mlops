We have made two different version to test data drift as the targets are a bit tricky to work with
as it is multi-label classification where more than one label can be chosen.

# 1 data_drift_single_label.py
In this script we combine every combination of the label into a single label, such that (0,0,0,0,0,0) -> 0, (0,0,0,0,0,1) -> 1, etc to
(1,1,1,1,1,1) -> 63.

# 2 data_drift_k_labels.py
In this script we do not combine and work directly on the 6 labels. However, this means that no "target" is specified and
the labels are then treated as features.

