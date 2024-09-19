#!/usr/bin/env python
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

from invoker import Script
from util.mpl import configure_mpl


def apparent2actual(vs, alpha_s, vo, alpha_o):
    a = vo * np.sin(alpha_o / 180 * np.pi) - vs * np.sin(alpha_s / 180 * np.pi)
    b = vo * np.cos(alpha_o / 180 * np.pi) - vs * np.cos(alpha_s / 180 * np.pi)
    beta_o = np.arctan2(a, b)
    return b / np.cos(beta_o), beta_o / np.pi * 180


def actual2apparent(vs, alpha_s, wo, beta_o):
    a = wo * np.sin(beta_o / 180 * np.pi) + vs * np.sin(alpha_s / 180 * np.pi)
    b = wo * np.cos(beta_o / 180 * np.pi) + vs * np.cos(alpha_s / 180 * np.pi)
    alpha_o = np.arctan2(a, b)
    return b / np.cos(alpha_o), alpha_o / np.pi * 180


class PlotApplicationResults(Script):
    @classmethod
    def args(cls):
        args = super().args()
        args.update(dict(
            # Specify arguments to pass from command line
            data_paths                   = [],
            model_data_path              = "./io/data/fullv2/agg/fullv2_all.csv",
            treatment_group_colnames     = ["TrialConfig.ScenePerlinSurfaceConfig.ContainerConfig.MovementSpeed",
                                            "TrialConfig.MetaConfig.Condition.Scene.View.Heading",
                                            "TrialConfig.MetaConfig.Condition.Target.SurfaceOffset"],
            output_path                  = "./io/figures/default.pdf",
            condition_name               = "default",
            condition_perceived_heading  = [ 0.0, 0.0, 0.0],
            rng_seed                     = 1,
            skip_write                   = False,
            display                      = False,
        ))
        return args

    @classmethod
    def modules(cls):
        mods = super().modules()
        mods.update(dict(
            # Add module dependencies
            motion_model        = "no_cross",
            study_config_parser = "base",
        ))
        return mods

    def run(self):

        rng = np.random.default_rng(self.opt.rng_seed)

        logging.info("Running script PlotApplicationResults")

        # Model Load
        df = pd.read_csv(self.opt.model_data_path)
        df_dict = self.study_config_parser.split_df_by_groups(df, self.opt.treatment_group_colnames)

        speed_data, heading_data, height_ratio_data = self.motion_model.load_data(df, df_dict)
        self.motion_model.optimize_model(speed_data, heading_data, height_ratio_data)

        if self.opt.condition_name == "sports":
            condition_labels = ["Control", "Camera Pose", "Camera Pose +\nScene Content"]
            speeds = np.array([1, 1, 1])
            headings = np.array([25, 25, 25])
            hratios = np.array([0.7, 0.6, 0.1])
        elif self.opt.condition_name == "flight":
            condition_labels = ["Control", "Static Scene", "Dynamic Scene"]
            speeds = np.array([0.5, 0.5, 0.5])
            headings = np.array([25, 25, 37])
            hratios = np.array([0.8, 0.4, 0.4])
        else:
            raise NotImplementedError()

        predict = self.motion_model(speeds, headings, hratios)
        perceived_heading = predict[:, 0] + headings
        perceived_std = predict[:, 1]
        logging.info("Predicted perceived headings: %.3f, %.3f, %.3f", *perceived_heading)

        # Data Load
        user_response_arrs = []
        for data_path in self.opt.data_paths:
            user_response_arr = np.loadtxt(data_path, dtype=int, delimiter=",")
            user_response_arrs.append(user_response_arr)
        freq_arr = np.stack(user_response_arrs)
        prob_arr = freq_arr / freq_arr.sum(axis=-1, keepdims=True)
        heading_arr = np.linspace(-30, 30, 7)

        # Stats
        mean_response = prob_arr @ heading_arr

        subjects = np.arange(mean_response.shape[0])
        conditions = ["A", "B", "C"]
        long_format_data = []
        for i, subject, in enumerate(subjects):
            for j, condition in enumerate(conditions):
                long_format_data.append([subject, condition, mean_response[i, j]])
        df = pd.DataFrame(long_format_data, columns=["Subject", "Condition", "Value"])

        aovrm = AnovaRM(df, "Value", "Subject", within=["Condition"])
        res = aovrm.fit()

        logging.info("Anova Results:\n%s", res) 

        # Raw responses
        mean_lines = mean_response.mean(axis=0)

        logging.info("Mean responses %.3f, %.3f, %.3f", *mean_lines)
        logging.info("SEM responses %.3f, %.3f, %.3f", *stats.sem(mean_response))

        plt.figure(figsize=(5,2))
        if self.opt.condition_name == "sports":
            ax = plt.axes([0.3, 0.025, 0.675, 0.700/3*4])
            ax.tick_params(direction="in", labelbottom=False, bottom=False, labelleft=True, left=False)
            ax.set_ylim(0, 4)
        else:
            ax = plt.axes([0.3, 0.275, 0.675, 0.700])
            ax.tick_params(direction="in", labelbottom=True, bottom=True, labelleft=True, left=False)
            ax.set_ylim(0, 3)
            ax.set_xticks([30, 20, 10, 0, -10, -20, -30])
            ax.set_xticklabels(["30°", "20°", "10°", "0°", "-10°", "-20°", "-30°"])
            ax.set_xlabel(r"Scene-Relative Target Heading, $\psi_t$")
        CONTROL_Y, TREATMENTA_Y, TREATMENTB_Y = 2.5, 1.5, 0.5
        ax.set_yticks([CONTROL_Y, TREATMENTA_Y, TREATMENTB_Y])
        ax.set_yticklabels(condition_labels)
        ax.set_xlim(35, -35)

        ax.fill_between([-35, -25], [0, 0], [4, 4], color="lightgray", alpha=0.5, zorder=0)
        ax.fill_between([-15,  -5], [0, 0], [4, 4], color="lightgray", alpha=0.5, zorder=0)
        ax.fill_between([  5,  15], [0, 0], [4, 4], color="lightgray", alpha=0.5, zorder=0)
        ax.fill_between([ 25,  35], [0, 0], [4, 4], color="lightgray", alpha=0.5, zorder=0)

        LW = 3.0
        SIMULATED_N = 22

        ax.plot([-20, -20], [0, 3], color="gray", linewidth=LW, zorder=1, linestyle="dotted")

        # CONTROL
        control_x = mean_response[:, 0]
        control_y = rng.uniform(-0.2, 0.2, size=control_x.shape) + CONTROL_Y
        ax.scatter(control_x, control_y, color="#fb5607", alpha=0.5, zorder=1)

        headings = np.linspace(-30, 30, 1000)
        probs = 1 / (perceived_std[0] * np.sqrt(2)) * np.exp(-(headings - perceived_heading[0]) ** 2 / np.sqrt(2 * perceived_std[0] ** 2))
        actual_speed, actual_angle = apparent2actual(2, headings, *actual2apparent(2, 25, 1, -20))

        simulated_x = actual_angle[rng.choice(probs.size, SIMULATED_N, p=probs/probs.sum())]
        simulated_y = rng.uniform(-0.2, 0.2, size=simulated_x.shape) + CONTROL_Y
        ax.scatter(simulated_x, simulated_y, color="black", alpha=0.5, zorder=0)

        actual_speed, actual_angle = apparent2actual(2, perceived_heading[0], *actual2apparent(2, 25, 1, -20))
        logging.info("Control condition predicted angle: %.3f", actual_angle)

        # TREATMENT A
        treatmenta_x = mean_response[:, 1]
        treatmenta_y = rng.uniform(-0.2, 0.2, size=treatmenta_x.shape) + TREATMENTA_Y
        ax.scatter(treatmenta_x, treatmenta_y, color="#ffbe0b", alpha=0.5, zorder=1)

        headings = np.linspace(-30, 30, 1000)
        probs = 1 / (perceived_std[1] * np.sqrt(2)) * np.exp(-(headings - perceived_heading[1]) ** 2 / np.sqrt(2 * perceived_std[1] ** 2))
        actual_speed, actual_angle = apparent2actual(2, headings, *actual2apparent(2, 25, 1, -20))

        simulated_x = actual_angle[rng.choice(probs.size, SIMULATED_N, p=probs/probs.sum())]
        simulated_y = rng.uniform(-0.2, 0.2, size=simulated_x.shape) + TREATMENTA_Y
        ax.scatter(simulated_x, simulated_y, color="black", alpha=0.5, zorder=0)

        actual_speed, actual_angle = apparent2actual(2, perceived_heading[1], *actual2apparent(2, 25, 1, -20))
        logging.info("Treatment A condition predicted angle: %.3f", actual_angle)

        # TREATMENT B
        treatmentb_x = mean_response[:, 2]
        treatmentb_y = rng.uniform(-0.2, 0.2, size=treatmentb_x.shape) + TREATMENTB_Y
        ax.scatter(treatmentb_x, treatmentb_y, color="#5dc133", alpha=0.5, zorder=1)

        headings = np.linspace(-30, 30, 1000)
        probs = 1 / (perceived_std[2] * np.sqrt(2)) * np.exp(-(headings - perceived_heading[2]) ** 2 / np.sqrt(2 * perceived_std[2] ** 2))
        actual_speed, actual_angle = apparent2actual(2, headings, *actual2apparent(2, 25, 1, -20))

        simulated_x = actual_angle[rng.choice(probs.size, SIMULATED_N, p=probs/probs.sum())]
        simulated_y = rng.uniform(-0.2, 0.2, size=simulated_x.shape) + TREATMENTB_Y
        ax.scatter(simulated_x, simulated_y, color="black", alpha=0.5, zorder=0)

        actual_speed, actual_angle = apparent2actual(2, perceived_heading[2], *actual2apparent(2, 25, 1, -20))
        logging.info("Treatment B condition predicted angle: %.3f", actual_angle)

        if not self.opt.skip_write:
            plt.savefig(self.opt.output_path)
            logging.info("Saved Figure: %s", self.opt.output_path)

        if self.opt.display:
            plt.show()


if __name__ == "__main__":
    configure_mpl()
    PlotApplicationResults().initialize().run()
