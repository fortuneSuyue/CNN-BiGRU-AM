import numpy as np
import random
import os


# https://zhuanlan.zhihu.com/p/104019160
def init_random_seed(seed=256, cudnnDeterministic=True, cudnnBenchmark=False, using_torch=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if using_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 为了保证可复现性, defaul True False; False True 可能可以提升gpu运行效率
        torch.backends.cudnn.deterministic = cudnnDeterministic
        torch.backends.cudnn.benchmark = cudnnBenchmark


def init_matplotlib_style(style='science'):
    """
    Use Matplotlib style settings from a style specification.
    Style files are stored in  C:\\Users\\L\\.matplotlib\\stylelib.

    They can be shown with 'import matplotlib.pyplot as plt | print(plt.style.available)'.

    The style name of 'default' is reserved for reverting back to
    the default style settings.

    .. note::

       This updates the `.rcParams` with the settings from the style.
       `.rcParams` not defined in the style are kept.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see `style.available`.       |
        |      | Recommendation: 'classic', 'matlab', 'nature', 'science',   |
        |      | 'ieee', 'ggplot', 'latex-sans','seaborn', 'seaborn-paper',  |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | `matplotlib.rcParams`.                                      |
        +------+-------------------------------------------------------------+
        | Path | A path-like object which is a path to a style file.         |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str, Path or dict) applied from |
        |      | first to last in the list.                                  |
        +------+-------------------------------------------------------------+
    """

    import matplotlib.pyplot as plt
    import matplotlib

    plt.style.use(style=style)
    matplotlib.rcParams['text.usetex'] = False  # LaTex style
