#!/usr/bin/env python
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf

from invoker import Script
from util.mpl import configure_mpl


class PlotPsychometricHeatmap(Script):
    @classmethod
    def args(cls):
        args = super().args()
        args.update(dict(
            speed_figure_path            = "./io/figures/fullv2/speed-pc.pdf",
            heading_figure_path          = "./io/figures/fullv2/heading-pc.pdf",
            hratio_figure_path           = "./io/figures/fullv2/hratio-pc.pdf",
            data_path                    = "./io/data/fullv2/agg/fullv2_all.csv",
            treatment_group_colnames     = ["TrialConfig.ScenePerlinSurfaceConfig.ContainerConfig.MovementSpeed",
                                            "TrialConfig.MetaConfig.Condition.Scene.View.Heading",
                                            "TrialConfig.MetaConfig.Condition.Target.SurfaceOffset"],
            fontsize                     = 10,
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
        logging.info("Running script PlotPsychometricHeatmap")

        df = pd.read_csv(self.opt.data_path)
        df_dict = self.study_config_parser.split_df_by_groups(df, self.opt.treatment_group_colnames)

        speed_data, heading_data, hratio_data = self.motion_model.load_data(df, df_dict)
        self.motion_model.optimize_model(speed_data, heading_data, hratio_data)

        CONTROL_SPEED = 1.0
        CONTROL_HEADING = 15.0
        CONTROL_HRATIO = 1.23 / (1.23 + 6 * np.sin(np.deg2rad(5)))

        def cdf(x, mean, std, lapse_rate):
            return lapse_rate + (1 - 2 * lapse_rate) * 1/2 * (1 + erf((x - mean)/(np.sqrt(2) * std)))

        # SPEED PLOT
        x = np.logspace(-0.42, 0.55, 101)
        y = np.linspace(0, 20, 101)
        xx, yy = np.meshgrid(x, y)

        model_inp_speed = xx.flatten()
        model_prediction = self.motion_model(model_inp_speed,
                                             np.ones_like(model_inp_speed) * CONTROL_HEADING,
                                             np.ones_like(model_inp_speed) * CONTROL_HRATIO,
                                             heading_override=True,
                                             hratio_override=True)
        mean = model_prediction[:, 0].reshape(xx.shape)
        std = model_prediction[:, 1].reshape(xx.shape)

        probability_map = cdf(yy, mean+15, std, np.ones_like(xx) * self.motion_model.lapse_rate)

        plt.figure(figsize=(5, 4.5))
        ax = plt.axes([0.125, 0.075, 0.850, 0.900])

        plt.contourf(xx, yy, probability_map, levels=np.linspace(0, 1, 100), cmap="plasma")

        cs = plt.contour(xx, yy, probability_map,
                         levels=[0.022, 0.089, 0.25, 0.5, 0.75, 0.911, 0.978],
                         colors=["white", "white", "white", "white", "black", "black", "black"],
                         linewidths=[0.5, 0.5, 0.5, 3, 0.5, 0.5, 0.5],
                         linestyles="solid")
        ax.clabel(cs, cs.levels, inline=True, fmt=lambda x: rf"{int(round(x*100))} \%", fontsize=self.opt.fontsize)

        ax.plot([10**-0.42, 10**0.55], [15, 15], linestyle="dotted", color="black", lw=3)

        ax.errorbar(speed_data[:, 0], speed_data[:, 3]+15, yerr=speed_data[:, 4]*0.674, marker='o', color="white", capsize=4, ls='none')

        logging.info("Speed conditions: mean (%s), std (%s)", speed_data[:, 3] + 15, speed_data[:, 4])

        ax.set_xscale("log")
        plt.minorticks_off()
        ax.set_xticks([0.5, 1.0, 3.0])
        ax.set_xticklabels(["0.5 m/s", "1 m/s", "3 m/s"])
        ax.set_ylim(0, 20)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_yticklabels(["0°", "5°", "10°", "15°", "20°"])
        ax.set_ylabel(r"Target Heading, $\varphi_t$")

        if not self.opt.skip_write:
            speed_figure_path = Path(self.opt.speed_figure_path)
            speed_figure_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(speed_figure_path)
            logging.info("Saved Figure: %s", speed_figure_path)

        # HEADING PLOT
        x = np.linspace(0, 26, 101)
        y = np.linspace(0, 20, 101)
        xx, yy = np.meshgrid(x, y)

        model_inp_heading = xx.flatten()
        model_prediction = self.motion_model(np.ones_like(model_inp_heading) * CONTROL_SPEED,
                                             model_inp_heading,
                                             np.ones_like(model_inp_heading) * CONTROL_HRATIO,
                                             speed_override=True,
                                             hratio_override=True)
        mean = model_prediction[:, 0].reshape(xx.shape)
        std = model_prediction[:, 1].reshape(xx.shape)

        probability_map = cdf(yy, mean+xx, std, np.ones_like(xx) * self.motion_model.lapse_rate)

        plt.figure(figsize=(4.5, 4.5))
        ax = plt.axes([0.025, 0.075, 0.950, 0.900])

        plt.contourf(xx, yy, probability_map, levels=np.linspace(0, 1, 100), cmap="plasma")

        cs = plt.contour(xx, yy, probability_map,
                         levels=[0.022, 0.089, 0.25, 0.5, 0.75, 0.911, 0.978],
                         colors=["white", "white", "white", "white", "black", "black", "black"],
                         linewidths=[0.5, 0.5, 0.5, 3, 0.5, 0.5, 0.5],
                         linestyles="solid")
        ax.clabel(cs, cs.levels, inline=True, fmt=lambda x: rf"{int(round(x*100))} \%", fontsize=self.opt.fontsize)

        ax.plot([0, 20], [0, 20], linestyle="dotted", color="black", lw=3)

        ax.errorbar(heading_data[:, 1], heading_data[:, 3] + heading_data[:, 1], yerr=heading_data[:, 4]*0.674, marker='o', color="white", capsize=4, ls='none')

        logging.info("Heading conditions: mean (%s), std (%s)", heading_data[:, 3] + heading_data[:, 1], heading_data[:, 4])

        ax.set_xticks([0, 5, 15, 25])
        ax.set_xticklabels(["0°", "5°", "15°", "25°"])
        ax.set_ylim(0, 20)
        ax.tick_params(left=False, labelleft=False)

        if not self.opt.skip_write:
            heading_figure_path = Path(self.opt.heading_figure_path)
            heading_figure_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(heading_figure_path)
            logging.info("Saved Figure: %s", heading_figure_path)

        # HEIGHT PLOT
        x = np.linspace(0,  1, 101)
        y = np.linspace(0, 20, 101)
        xx, yy = np.meshgrid(x, y)

        model_inp_hratio = xx.flatten()
        model_prediction = self.motion_model(np.ones_like(model_inp_hratio) * CONTROL_SPEED,
                                             np.ones_like(model_inp_hratio) * CONTROL_HEADING,
                                             model_inp_hratio,
                                             speed_override=True,
                                             heading_override=True)
        mean = model_prediction[:, 0].reshape(xx.shape)
        std = model_prediction[:, 1].reshape(xx.shape)

        probability_map = cdf(yy, mean+15, std, np.ones_like(xx) * self.motion_model.lapse_rate)

        plt.figure(figsize=(4.5, 4.5))
        ax = plt.axes([0.025, 0.075, 0.950, 0.900])

        plt.contourf(xx, yy, probability_map, levels=np.linspace(0, 1, 100), cmap="plasma")

        cs = plt.contour(xx, yy, probability_map,
                         levels=[0.022, 0.089, 0.25, 0.5, 0.75, 0.911, 0.978],
                         colors=["white", "white", "white", "white", "black", "black", "black"],
                         linewidths=[0.5, 0.5, 0.5, 3, 0.5, 0.5, 0.5],
                         linestyles="solid")
        ax.clabel(cs, cs.levels, inline=True, fmt=lambda x: rf"{int(round(x*100))} \%", fontsize=self.opt.fontsize)

        ax.plot([0, 1], [15, 15], linestyle="dotted", color="black", lw=3)

        ax.errorbar(hratio_data[:, 2], hratio_data[:, 3]+15, yerr=hratio_data[:, 4]*0.674, marker='o', color="white", capsize=4, ls='none')

        logging.info("Hratio conditions: mean (%s), std (%s)", hratio_data[:, 3] + hratio_data[:, 1], hratio_data[:, 4])

        heights = np.array([4.7, 1.23, 0.22, 0.03])
        hratios = heights / (heights+6*np.sin(np.deg2rad(5)))
        ax.set_xticks(hratios)
        ax.set_xticklabels([f"{hratio:.2f}" for hratio in hratios])
        ax.set_ylim(0, 20)
        ax.tick_params(left=False, labelleft=False)

        if not self.opt.skip_write:
            hratio_figure_path = Path(self.opt.hratio_figure_path)
            hratio_figure_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(hratio_figure_path)
            logging.info("Saved Figure: %s", hratio_figure_path)

        if self.opt.display:
            plt.show()


if __name__ == "__main__":
    configure_mpl()
    PlotPsychometricHeatmap().initialize().run()
