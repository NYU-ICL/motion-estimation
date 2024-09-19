#!/usr/bin/env python
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from invoker import Script
from util.mpl import configure_mpl


class PlotMotionModel(Script):
    @classmethod
    def args(cls):
        args = super().args()
        args.update(dict(
            mean_figure_path             = "./io/figures/fullv2/mean-model.pdf",
            std_figure_path              = "./io/figures/fullv2/std-model.pdf",
            data_path                    = "./io/data/fullv2/agg/fullv2_all.csv",
            treatment_group_colnames     = ["TrialConfig.ScenePerlinSurfaceConfig.ContainerConfig.MovementSpeed",
                                            "TrialConfig.MetaConfig.Condition.Scene.View.Heading",
                                            "TrialConfig.MetaConfig.Condition.Target.SurfaceOffset"],
            skip_write                   = False,
            display                      = False,
        ))
        return args

    @classmethod
    def modules(cls):
        mods = super().modules()
        mods.update(dict(
            motion_model        = "no_cross",
            study_config_parser = "base",
        ))
        return mods

    def run(self):
        logging.info("Running script PlotMotionModel")

        df = pd.read_csv(self.opt.data_path)
        df_dict = self.study_config_parser.split_df_by_groups(df, self.opt.treatment_group_colnames)

        speed_data, heading_data, height_ratio_data = self.motion_model.load_data(df, df_dict)
        self.motion_model.optimize_model(speed_data, heading_data, height_ratio_data)

        heading_range = np.linspace(5, 25, 50)
        height_range = np.linspace(0.05, 0.9, 50)
        xx, yy = np.meshgrid(heading_range, height_range)
        predict_lo = self.motion_model(np.ones_like(xx).flatten() * 0.5, xx.flatten(), yy.flatten()).reshape(*xx.shape, -1)
        predict_hi = self.motion_model(np.ones_like(xx).flatten() * 3.0, xx.flatten(), yy.flatten()).reshape(*xx.shape, -1)

        mean_lo, std_lo = predict_lo[..., 0] + xx, predict_lo[..., 1]
        mean_hi, std_hi = predict_hi[..., 0] + xx, predict_hi[..., 1]

        plt.figure(figsize=(5, 5))
        ax = plt.axes(projection="3d")
        ax.view_init(30, 50)

        norm = plt.Normalize(min(mean_hi.min(), mean_lo.min()), max(mean_hi.max(), mean_lo.max()))

        for _ in range(2):
            ax.plot_surface(xx, yy, mean_hi, rstride=1, cstride=1, cmap=cm.plasma, norm=norm, linewidths=0, antialiased=False)
        ax.plot_wireframe(xx, yy, mean_hi, rstride=10, cstride=10, color=(1.0, 1.0, 1.0, 0.1))

        for _ in range(2):
            ax.plot_surface(xx, yy, mean_lo, rstride=1, cstride=1, cmap=cm.plasma, norm=norm, linewidths=0, antialiased=False)
        ax.plot_wireframe(xx, yy, mean_lo, rstride=10, cstride=10, color=(1.0, 1.0, 1.0, 0.1))

        ax.xaxis.set_pane_color((1, 1, 1, 1))
        ax.yaxis.set_pane_color((1, 1, 1, 1))
        ax.zaxis.set_pane_color((1, 1, 1, 1))
        lblx = ax.set_xlabel(r"Scene Heading, $\varphi_s$")
        lbly = ax.set_ylabel(r"Depth Disparity, $d$")

        ax.set_xticks([5, 15, 25])
        ax.set_xticklabels(["5°", "15°", "25°"])
        ax.set_yticks([0.00, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticks([5, 10, 15, 20])
        ax.set_zticklabels(["5°", "10°", "15°", "20°"])
        ax.set_xlim3d(5, 25)
        ax.set_ylim3d(0, 1)
        ax.autoscale(enable=True, axis="both", tight=False)

        if not self.opt.skip_write:
            plt.savefig(self.opt.mean_figure_path, bbox_extra_artists=(lblx, lbly))
            logging.info("Saved Figure: %s", self.opt.mean_figure_path)

        plt.figure(figsize=(5, 5))
        ax = plt.axes(projection="3d")
        ax.view_init(30, 50)

        norm = plt.Normalize(min(std_hi.min(), std_lo.min()), max(std_hi.max(), std_lo.max()))

        for _ in range(2):
            ax.plot_surface(xx, yy, std_hi, rstride=1, cstride=1, cmap=cm.plasma, norm=norm, linewidths=0, antialiased=False)
        ax.plot_wireframe(xx, yy, std_hi, rstride=10, cstride=10, color=(1.0, 1.0, 1.0, 0.1))

        for _ in range(2):
            ax.plot_surface(xx, yy, std_lo, rstride=1, cstride=1, cmap=cm.plasma, norm=norm, linewidths=0, antialiased=False)
        ax.plot_wireframe(xx, yy, std_lo, rstride=10, cstride=10, color=(1.0, 1.0, 1.0, 0.1))

        ax.xaxis.set_pane_color((1, 1, 1, 1))
        ax.yaxis.set_pane_color((1, 1, 1, 1))
        ax.zaxis.set_pane_color((1, 1, 1, 1))
        lblx = ax.set_xlabel(r"Scene Heading, $\varphi_s$")
        lbly = ax.set_ylabel(r"Depth Disparity, $d$")

        ax.set_xticks([5, 15, 25])
        ax.set_xticklabels(["5°", "15°", "25°"])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticks([4, 6, 8, 10])
        ax.set_zticklabels(["4°", "6°", "8°", "10°"])
        ax.set_xlim3d(5, 25)
        ax.set_ylim3d(0, 1)
        ax.autoscale(enable=True, axis="both", tight=False)

        if not self.opt.skip_write:
            plt.savefig(self.opt.std_figure_path, bbox_extra_artists=(lblx, lbly))
            logging.info("Saved Figure: %s", self.opt.std_figure_path)

        if self.opt.display:
            plt.show()


if __name__ == "__main__":
    configure_mpl()
    PlotMotionModel().initialize().run()
