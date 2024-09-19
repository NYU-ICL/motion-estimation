#!/usr/bin/env python
import logging

import numpy as np
import pandas as pd
from scipy.special import erf

from invoker import Script
from util.mpl import configure_mpl


class AnalyzeModelFit(Script):
    @classmethod
    def args(cls):
        args = super().args()
        args.update(dict(
            model_data_paths             = [""],
            eval_data_paths              = [""],
            treatment_group_colnames     = ["TrialConfig.ScenePerlinSurfaceConfig.ContainerConfig.MovementSpeed",
                                            "TrialConfig.MetaConfig.Condition.Scene.View.Heading",
                                            "TrialConfig.MetaConfig.Condition.Target.SurfaceOffset"],
            rng_seed                     = 1,
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
        logging.info("Running script AnalyzeModelFit")

        def load(data_path):
            df = pd.read_csv(data_path)
            df_dict = self.study_config_parser.split_df_by_groups(df, self.opt.treatment_group_colnames)
            return df, df_dict

        def cdf(x, mean, std, lapse_rate):
            return lapse_rate + (1 - 2 * lapse_rate) * 1/2 * (1 + erf((x - mean)/(np.sqrt(2) * std)))

        def rsq(data, pred):
            return 1 - np.sum((data - pred)**2) / np.sum((data - np.mean(data))**2)

        rsq_grid = []
        for i, (model_data_path, eval_data_path) in enumerate(zip(self.opt.model_data_paths, self.opt.eval_data_paths)):
            logging.info("Evaluating model fit on %s against data of %s", model_data_path, eval_data_path)

            speed_data, heading_data, hratio_data = self.motion_model.load_data(*load(model_data_path))
            self.motion_model.optimize_model(speed_data, heading_data, hratio_data)

            eval_df, eval_df_dict = load(eval_data_path)
            eval_speed_data, eval_heading_data, eval_hratio_data = self.motion_model.load_data(eval_df, eval_df_dict)

            eval_inp = np.concatenate([eval_speed_data[:, :3], eval_heading_data[:, :3], eval_hratio_data[:, :3]])
            eval_data = np.concatenate([eval_speed_data[:, 3:], eval_heading_data[:, 3:], eval_hratio_data[:, 3:]])
            eval_pred = self.motion_model(eval_inp[:, 0], eval_inp[:, 1], eval_inp[:, 2])

            rsq_arr = []
            for (speed, heading, height), eval_df in eval_df_dict.items():

                data_x = (eval_df["TrialConfig.MetaConfig.StimulusLevel.BiasPercentage"] / 100 * heading).to_numpy()
                data_prob = eval_df["TrialResponse.AgainstResponseProbability.mean"].to_numpy()

                hratio = height / (height + 6 * np.sin(np.deg2rad(5)))
                model_prediction = self.motion_model(np.array([speed]), np.array([heading]), np.array([hratio]))
                mean = model_prediction[:, 0]
                std = model_prediction[:, 1]
                lapse_rate = self.motion_model.lapse_rate

                pred_prob = cdf(data_x, mean, std, lapse_rate)
                r_squared = rsq(data_prob, pred_prob)
                rsq_arr.append(r_squared)

                logging.info("Condition: %s, r_squared: %.3f", [speed, heading, height], r_squared)

            rsq_grid.append(np.array(rsq_arr))

        rsq_grid = np.array(rsq_grid)

        logging.info("Overall mean rsq: %.3f, min rsq: %.3f", rsq_grid.mean(), rsq_grid.min())



if __name__ == "__main__":
    configure_mpl()
    AnalyzeModelFit().initialize().run()
