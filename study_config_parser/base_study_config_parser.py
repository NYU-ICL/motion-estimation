import xml.etree.ElementTree as ET

import pandas as pd

from invoker import Module


class BaseStudyConfigParser(Module):
    @classmethod
    def args(cls):
        args = super().args()
        args.update(dict(
            # Specify arguments to pass from command line
            rng_seed=0,
        ))
        return args

    VECTOR2_SCHEMA = dict(x=float, y=float)
    VECTOR3_SCHEMA = dict(x=float, y=float, z=float)
    QUATERNION_SCHEMA = dict(x=float, y=float, z=float, w=float)

    META_CONFIG_SCHEMA = dict(
        Condition=dict(
            Target=dict(
                View=dict(
                    Eccentricity=float,
                    Azimuth=float,
                    Depth=float,
                ),
                SurfaceOffset=float,
            ),
            Scene=dict(
                View=dict(
                    Heading=float,
                    Elevation=float,
                ),
            ),
            CameraViewElevation=float,
        ),
        StimulusLevel=dict(
            Unbiased=float,
            Bias=float,
            Computed=float,
        ),
    )

    VOLUMETRIC_PARTICLE_SYSTEM_SCHEMA = dict(
        ContainerConfig=dict(
            Position=VECTOR3_SCHEMA,
            Quaternion=QUATERNION_SCHEMA,
            MovementDirection=VECTOR3_SCHEMA,
            MovementSpeed=float,
        ),
        ParticleConfig=dict(
            Density=float,
            Speed=float,
        ),
    )

    PERLIN_SURFACE_SCHEMA = dict(
        ContainerConfig=dict(
            Position=VECTOR3_SCHEMA,
            Quaternion=QUATERNION_SCHEMA,
            MovementDirection=VECTOR3_SCHEMA,
            MovementSpeed=float,
        ),
        TextureConfig=dict(
            Heading=VECTOR3_SCHEMA,
            Speed=float,
        ),
    )

    TRIAL_CONFIG_SCHEMA = dict(
        Duration=float,
        MetaConfig=META_CONFIG_SCHEMA,
        TargetConfig=VOLUMETRIC_PARTICLE_SYSTEM_SCHEMA,
        ScenePerlinSurfaceConfig=PERLIN_SURFACE_SCHEMA,
    )

    OLD_TRIAL_CONFIG_SCHEMA = dict(
        Duration=float,
        SceneParticleSpeed=float,
        SceneParticleDensity=float,
        SceneHeadingDirection=dict(r=float, p=float, y=float),
        ObjectParticleSpeed=float,
        ObjectParticleDensity=float,
        ObjectDirectionAngle=float,
        ObjectDirectionVector=dict(x=float, y=float, z=float),
        ObjectSpeed=float,
        ObjectOrigin=dict(x=float, y=float, z=float),
    )

    TRIAL_RESPONSE_SCHEMA = dict(
        ResponseKey=str,
        ResponseResult=str,
    )

    TRIAL_LOG_SCHEMA = dict(
        TrialId=str,
        TrialConfig=TRIAL_CONFIG_SCHEMA,
        TrialResponse=TRIAL_RESPONSE_SCHEMA,
    )

    def load_block_log(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        trial_logs = []
        for trial_log in root.iter("TrialLog"):
            trial_log_dict = _parse_etree_by_schema(trial_log, self.TRIAL_LOG_SCHEMA)
            trial_log_dict = _flatten_dict(trial_log_dict)
            trial_logs.append(trial_log_dict)
        df = pd.DataFrame(trial_logs)
        return df

    def split_df_by_groups(self, df, groupby_colnames):
        if len(groupby_colnames) == 0:
            return [df]
        grouped_df = df.groupby(groupby_colnames)
        split_df_dict = {g: grouped_df.get_group(g) for g in grouped_df.groups}
        # TODO: bad behavior when "g" is not a tuple
        return split_df_dict

    def sample_df_by_group(self, df, groupby_colnames, nrows):
        return df.groupby(groupby_colnames)\
                 .sample(n=nrows, random_state=self.opt.rng_seed)

    def sample_df_groups(self, df, group_name, ngroups):
        stimulus_values = sorted(df[group_name].unique())
        stimulus_values = stimulus_values[::len(stimulus_values) // ngroups]
        return df[df[group_name].isin(stimulus_values)]


def _parse_etree_by_schema(node, schema):
    if type(schema) != dict and type(schema) != type:
        raise Exception("Unexpected node type %s" % schema)
    if type(schema) == type:
        return schema(node.text)
    node_dict = dict()
    for elabel, eschema in schema.items():
        # TODO: Make error handling more graceful
        try:
            element = node.find(elabel)
        except AttributeError:
            raise Exception(f"Could not find element {elabel} in schema {schema}")
        node_dict[elabel] = _parse_etree_by_schema(element, eschema)
    return node_dict


def _flatten_dict(data):
    output = dict()
    for k, v in data.items():
        if type(v) != dict:
            output[k] = v
            continue
        for ck, cv in _flatten_dict(v).items():
            output[f"{k}.{ck}"] = cv
    return output
