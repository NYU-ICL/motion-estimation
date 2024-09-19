#!/usr/bin/env python
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fit_psyche.psychometric_curve import PsychometricCurve

from invoker import Script
from util.mpl import configure_mpl


class PlotControlCondition(Script):
    @classmethod
    def args(cls):
        args = super().args()
        args.update(dict(
            data_path                    = "./io/data/fullv2/agg/fullv2_all.csv",
            figure_root                  = "./io/figures/fullv2/",
            figure_fname                 = "psychcurve-control.pdf",
            treatment_group_colnames     = ["TrialConfig.ScenePerlinSurfaceConfig.ContainerConfig.MovementSpeed",
                                            "TrialConfig.MetaConfig.Condition.Scene.View.Heading",
                                            "TrialConfig.MetaConfig.Condition.Target.SurfaceOffset"],
            stimulus_value_colname       = "TrialConfig.MetaConfig.StimulusLevel.BiasPercentage",
            response_prob_mean_colname   = "TrialResponse.AgainstResponseProbability.mean",
            response_prob_sem_colname    = "TrialResponse.AgainstResponseProbability.sem",
            pc_mean_lims                 = [-220, 220],
            pc_std_lims                  = [0, 200],
            interp_nsamples              = 40,
            skip_write                   = False,
            display                      = False,
        ))
        return args

    @classmethod
    def modules(cls):
        mods = super().modules()
        mods.update(dict(
            # Add module dependencies
            study_config_parser = "base",
        ))
        return mods

    def run(self):
        logging.info("Running script PlotControlCondition")

        # Fetch control data
        df = pd.read_csv(self.opt.data_path)
        lapse_rate = 1 - df[df[self.opt.stimulus_value_colname] == 200][self.opt.response_prob_mean_colname].mean()
        logging.info("Lapse rate = %.3f", lapse_rate)

        df_dict = self.study_config_parser.split_df_by_groups(df, self.opt.treatment_group_colnames)
        speed, heading, surface_offset = 1.0, 15.0, 1.23
        df = df_dict[(speed, heading, surface_offset)]

        target_foe = df[self.opt.stimulus_value_colname] / 100. * heading
        x = target_foe.to_numpy().astype("float64")
        y = df[self.opt.response_prob_mean_colname]
        yerr = df[self.opt.response_prob_sem_colname]

        # Build psychometric curve
        pc_mean_lims = [lim / 100 * heading for lim in self.opt.pc_mean_lims]
        pc_std_lims = [lim / 100 * heading for lim in self.opt.pc_std_lims]

        pc = PsychometricCurve(model="wh",
                               mean_lims=pc_mean_lims,
                               var_lims=pc_std_lims,
                               guess_rate_lims=[lapse_rate, lapse_rate+1e-9],
                               lapse_rate_lims=[lapse_rate, lapse_rate+1e-9]).fit(x, y)
        logging.info("Psychometric parameters for control condition: mean (%.3f), std (%.3f), lapse_rate (%.3f)",
                     pc.coefs_["mean"], pc.coefs_["var"], pc.coefs_["lapse_rate"])

        soe_x = pc.coefs_["mean"]

        resampled_x = np.linspace(*pc_mean_lims, self.opt.interp_nsamples)
        resample_y = lambda pc: pc.predict(resampled_x)

        # Build Figure
        plt.figure(figsize=(4.5, 5.25))
        ax = plt.axes([0.025, 0.125, 0.950, 0.850])

        ax.tick_params(direction="in", left=False, labelleft=False, right=False, labelright=True, bottom=True, labelbottom=True)
        ax.tick_params(axis="y", pad=-65)
        plt.setp(ax.get_yticklabels(), va="bottom")

        ax.set_xlim(*pc_mean_lims)
        x_angles = np.array([-15, 0, 15, 30])
        ax.set_xticks(x_angles)
        ax.set_xticklabels(["0°\n.26 m/s", "15°\n0 m/s", "30°\n-.26 m/s", "45°\n-.5 m/s"])

        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["25\%", "50\%", "75\%"])

        ax.plot(pc_mean_lims, [0.5, 0.5], color="lightgray", linestyle="dotted", zorder=-1)
        ax.plot(pc_mean_lims, [0.25, 0.25], color="lightgray", linestyle="dotted", zorder=-1)
        ax.plot(pc_mean_lims, [0.75, 0.75], color="lightgray", linestyle="dotted", zorder=-1)

        ax.plot([soe_x, soe_x], [0, 1], color="black", linestyle="dotted", zorder=1)
        ax.errorbar(x, y, yerr=yerr, fmt=".", capsize=2, color="black", zorder=2)
        ax.plot(resampled_x, resample_y(pc), color="#06d6a0", label="$=\mu$", zorder=2)

        # I/O
        if not self.opt.skip_write:
            figure_root = Path(self.opt.figure_root)
            figure_root.mkdir(exist_ok=True, parents=True)
            figure_path = figure_root / self.opt.figure_fname

            plt.savefig(figure_path)
            logging.info("Saved Figure: %s", figure_path)

        if self.opt.display:
            plt.show()


if __name__ == "__main__":
    configure_mpl()
    PlotControlCondition().initialize().run()
