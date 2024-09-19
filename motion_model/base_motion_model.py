from invoker import Module


class BaseMotionModel(Module):
    @classmethod
    def args(cls):
        args = super().args()
        args.update(dict(
            # Specify arguments to pass from command line
        ))
        return args
