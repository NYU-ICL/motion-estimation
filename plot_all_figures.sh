#!/bin/bash

# Figure 2c
python plot_control_condition.py

# Section 3.1
Rscript analyze_mirrored_data.r

# Figure 5
python plot_psychometric_heatmap.py

# Figure 6
python plot_motion_model.py

# Section 4.2
python analyze_model_fit.py --model_data_paths ./io/data/fullv2/agg/fullv2_tophalf_0* --eval_data_paths ./io/data/fullv2/agg/fullv2_bothalf_0*

# Figure 9
python plot_application_results.py --data_paths ./io/data/eval_flight/agg/*.csv --output_path ./io/figures/eval/eval-results-flight.pdf --condition_name flight --rng_seed 2
python plot_application_results.py --data_paths ./io/data/eval_sports/agg/*.csv --output_path ./io/figures/eval/eval-results-sport.pdf --condition_name sports --rng_seed 6

# Figure 10
python plot_psychometric_heatmap.py --data_path ./io/data/fullv2_unfiltered/agg/fullv2_all.csv --speed_figure_path ./io/figures/fullv2_unfiltered/speed-pc-nofilter.pdf --heading_figure_path ./io/figures/fullv2_unfiltered/heading-pc-nofilter.pdf --hratio_figure_path ./io/figures/fullv2_unfiltered/hratio-pc-nofilter.pdf
