# Futurehuman - model testing report

## Run comparison (locked TEST split)

| run                   |   n |   macro_F1 | false_action              | missed_action           | unnecessary_ask        | schema_valid                |   ECE |   distinct_conf_values | pair_consistency       |   latency_p50_s |
|:----------------------|----:|-----------:|:--------------------------|:------------------------|:-----------------------|:----------------------------|------:|-----------------------:|:-----------------------|----------------:|
| base_A0_original_test |  83 |      0.469 | 40.0% [28.1-53.2] (22/55) | 14.3% [5.7-31.5] (4/28) | 0.0% [0.0-12.1] (0/28) | 97.6% [91.6-99.3] (81/83)   | 0.364 |                      4 | 0.0% [0.0-39.0] (0/6)  |            4.36 |
| base_A1_improved_test |  83 |      0.525 | 14.5% [7.6-26.2] (8/55)   | 14.3% [5.7-31.5] (4/28) | 0.0% [0.0-12.1] (0/28) | 100.0% [95.6-100.0] (83/83) | 0.16  |                      6 | 0.0% [0.0-39.0] (0/6)  |            3.98 |
| base_B_fewshot_test   |  83 |      0.638 | 21.8% [12.9-34.4] (12/55) | 0.0% [0.0-12.1] (0/28)  | 0.0% [0.0-12.1] (0/28) | 100.0% [95.6-100.0] (83/83) | 0.15  |                      5 | 16.7% [3.0-56.4] (1/6) |            3.54 |

## base_A0_original_test

Confusion matrix (rows = gold):

                actionable  non_actionable  ambiguous  invalid
actionable              24               4          0        0
non_actionable           6              22          6        0
ambiguous               16               4          1        0

Per class:

                precision  recall     f1  support
actionable          0.522   0.857  0.649     28.0
non_actionable      0.733   0.647  0.688     34.0
ambiguous           0.143   0.048  0.071     21.0

## base_A1_improved_test

Confusion matrix (rows = gold):

                actionable  non_actionable  ambiguous  invalid
actionable              24               4          0        0
non_actionable           1              33          0        0
ambiguous                7              14          0        0

Per class:

                precision  recall     f1  support
actionable          0.750   0.857  0.800     28.0
non_actionable      0.647   0.971  0.776     34.0
ambiguous           0.000   0.000  0.000     21.0

## base_B_fewshot_test

Confusion matrix (rows = gold):

                actionable  non_actionable  ambiguous  invalid
actionable              28               0          0        0
non_actionable           2              31          1        0
ambiguous               10               8          3        0

Per class:

                precision  recall     f1  support
actionable          0.700   1.000  0.824     28.0
non_actionable      0.795   0.912  0.849     34.0
ambiguous           0.750   0.143  0.240     21.0
