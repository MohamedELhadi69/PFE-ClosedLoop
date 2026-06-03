# Final Results

This folder contains the current clean outputs of the remediation pipeline.

- `llm_recommender_test_clean.csv`: student recommender outputs on the held-out test split. This is the main final CSV and includes `action_to_take`.
- `llm_reference_test_clean.csv`: reference decisions used to evaluate the student recommender.
- `llm_student_memory_clean.csv`: dataset-grounded memory built from the model-training split and used for retrieval during recommendation.
- `student_test_evaluation_clean/`: evaluation reports comparing the student outputs against the reference outputs.

Core input files such as `data.csv`, `remedy_catalogue_dataset.csv`, and the split index CSVs remain in the project root because the scripts expect them there.
