# Model

The base teacher is the full PC-STE v2 S1 model. The primary publication model is K1 (`half_4x1`): four forward-only temporal Mamba layers with `kept_direction=fwd`, `uni_residual=mean_of_remaining`, and 1,375,953 verified encoder parameters.

K1 shares one encoder across four datasets and retains dataset-specific heads. It is trained from the matched Full-S1 initialization with a same-fold, three-seed S1 teacher ensemble.

Knowledge distillation uses temperature 4, alpha 0.5, relational weight 1.0, `mean_prob_at_T` teacher aggregation, and KL direction teacher-to-student.

