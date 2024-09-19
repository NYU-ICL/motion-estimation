import logging

import numpy as np
from fit_psyche.psychometric_curve import PsychometricCurve
from sklearn.linear_model import LinearRegression

from .base_motion_model import BaseMotionModel


class NoCrossMotionModel(BaseMotionModel):
    @classmethod
    def args(cls):
        args = super().args()
        args.update(dict(
            # Specify arguments to pass from command line
            stimulus_value_colname       = "TrialConfig.MetaConfig.StimulusLevel.BiasPercentage",
            response_prob_mean_colname   = "TrialResponse.AgainstResponseProbability.mean",
            response_prob_sem_colname    = "TrialResponse.AgainstResponseProbability.sem",
            pc_mean_lims                 = [-200, 200],
            pc_std_lims                  = [0, 200],
            mean_deg                     = 2,
            std_deg                      = 1,
        ))
        return args

    def load_data(self, df, df_dict):
        lapse_rate = 1 - df[df[self.opt.stimulus_value_colname] == 200][self.opt.response_prob_mean_colname].mean()
        self.lapse_rate = lapse_rate
        logging.info("Lapse rate = %.3f", lapse_rate)

        speed_data = []
        heading_data = []
        hratio_data = []
        for (speed, heading, surface_offset), df in df_dict.items():
            target_foe = df[self.opt.stimulus_value_colname] / 100. * heading
            x = target_foe.to_numpy().astype("float64")
            y = df[self.opt.response_prob_mean_colname]

            pc_mean_lims = [lim / 100 * heading for lim in self.opt.pc_mean_lims]
            pc_std_lims = [lim / 100 * heading for lim in self.opt.pc_std_lims]

            pc = PsychometricCurve(model="wh",
                                   mean_lims=pc_mean_lims,
                                   var_lims=pc_std_lims,
                                   guess_rate_lims=[lapse_rate, lapse_rate+1e-9],
                                   lapse_rate_lims=[lapse_rate, lapse_rate+1e-9]).fit(x, y)

            camera_height = 6 * np.sin(np.deg2rad(5)) + surface_offset
            hratio = surface_offset / camera_height

            if speed == 1 and heading == 15:
                hratio_data.append([speed, heading, hratio, pc.coefs_["mean"], pc.coefs_["var"]])
            if heading == 15 and surface_offset == 1.23:
                speed_data.append([speed, heading, hratio, pc.coefs_["mean"], pc.coefs_["var"]])
            if speed == 1 and surface_offset == 1.23:
                heading_data.append([speed, heading, hratio, pc.coefs_["mean"], pc.coefs_["var"]])

            logging.info("%.3f, %.3f, %.3f: %3f, %3f", speed, heading, hratio, pc.coefs_["mean"], pc.coefs_["var"])

        speed_data = np.array(speed_data)
        heading_data = np.array(heading_data)
        hratio_data = np.array(hratio_data)

        return speed_data, heading_data, hratio_data

    def optimize_model(self, speed_data, heading_data, hratio_data):

        control_mean = speed_data[1:2, 3:4]
        control_std = speed_data[1:2, 4:5]

        logging.info("Control Condition: mean = %.3f, std = %.3f", control_mean[0, 0], control_std[0, 0])

        def build_polynomial_features(vals, n=2):
            if n == 1:
                return np.stack([vals], axis=-1)
            elif n == 2:
                return np.stack([vals, vals**2], axis=-1)
            else:
                raise NotImplementedError()

        def polynomial_logfmt(n):
            if n == 1:
                return "%.3f + %.3f x"
            elif n == 2:
                return "%.3f + %.3f x + %.3f x"
            else:
                raise NotImplementedError()

        speed_mean_reg = LinearRegression().fit(build_polynomial_features(speed_data[:, 0], n=self.opt.mean_deg), speed_data[:, 3:4] / control_mean)
        speed_std_reg  = LinearRegression().fit(build_polynomial_features(speed_data[:, 0], n=self.opt.std_deg),  speed_data[:, 4:5] / control_std)

        logging.info("Speed Regression: mean = %.3f + %.3f x + %.3f x**2", speed_mean_reg.intercept_[0], *speed_mean_reg.coef_[0])
        logging.info(f"Speed Regression: std = {polynomial_logfmt(self.opt.std_deg)}", speed_std_reg.intercept_[0], *speed_std_reg.coef_[0])

        heading_mean_reg = LinearRegression().fit(build_polynomial_features(heading_data[:, 1], n=self.opt.mean_deg), heading_data[:, 3:4] / control_mean)
        heading_std_reg  = LinearRegression().fit(build_polynomial_features(heading_data[:, 1], n=self.opt.std_deg),  heading_data[:, 4:5] / control_std)

        logging.info("Heading Regression: mean = %.3f + %.3f x + %.3f x**2", heading_mean_reg.intercept_[0], *heading_mean_reg.coef_[0])
        logging.info(f"Heading Regression: std = {polynomial_logfmt(self.opt.std_deg)}", heading_std_reg.intercept_[0], *heading_std_reg.coef_[0])

        hratio_mean_reg = LinearRegression().fit(build_polynomial_features(hratio_data[:, 2], n=self.opt.mean_deg), hratio_data[:, 3:4] / control_mean)
        hratio_std_reg  = LinearRegression().fit(build_polynomial_features(hratio_data[:, 2], n=self.opt.std_deg),  hratio_data[:, 4:5] / control_std)

        logging.info("Hratio Regression: mean = %.3f + %.3f x + %.3f x**2", hratio_mean_reg.intercept_[0], *hratio_mean_reg.coef_[0])
        logging.info(f"Hratio Regression: std = {polynomial_logfmt(self.opt.std_deg)}", hratio_std_reg.intercept_[0], *hratio_std_reg.coef_[0])

        def predict(speed, heading, hratio, speed_override=False, heading_override=False, hratio_override=False):
            speed_mean_k = speed_mean_reg.predict(build_polynomial_features(speed, n=self.opt.mean_deg))
            speed_std_k  = speed_std_reg.predict(build_polynomial_features(speed, n=self.opt.std_deg))

            heading_mean_k = heading_mean_reg.predict(build_polynomial_features(heading, n=self.opt.mean_deg))
            heading_std_k  = heading_std_reg.predict(build_polynomial_features(heading, n=self.opt.std_deg))

            hratio_mean_k = hratio_mean_reg.predict(build_polynomial_features(hratio, n=self.opt.mean_deg))
            hratio_std_k  = hratio_std_reg.predict(build_polynomial_features(hratio, n=self.opt.std_deg))

            if speed_override:
                speed_mean_k = np.ones_like(speed_mean_k)
                speed_std_k = np.ones_like(speed_std_k)
            if heading_override:
                heading_mean_k = np.ones_like(heading_mean_k)
                heading_std_k = np.ones_like(heading_std_k)
            if hratio_override:
                hratio_mean_k = np.ones_like(hratio_mean_k)
                hratio_std_k = np.ones_like(hratio_std_k)

            mean = speed_mean_k * heading_mean_k * hratio_mean_k * control_mean
            std = speed_std_k * heading_std_k * hratio_std_k * control_std
            out = np.concatenate([mean, std], axis=-1)
            return out

        self.model = predict

    def __call__(self, speed, heading, height_ratio, **kwargs):
        return self.model(speed, heading, height_ratio, **kwargs)
