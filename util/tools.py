
from neuralnets.util.tools import parse_params as parse_params_base


def parse_params(params):
    """
    Parse a YAML parameter dictionary

    :param params: dictionary containing the parameters
    :return: parsed dictionary
    """

    params = parse_params_base(params)

    keys = params.keys()

    if 'tar_labels_available' in keys:
        params['tar_labels_available'] = float(params['tar_labels_available'])

    for dom in ['src', 'tar']:
        if dom in keys:
            ks = params[dom].keys()
            if 'train_val_test_split' in ks:
                params[dom]['train_val_test_split'] = [float(item) for item in
                                                       params[dom]['train_val_test_split'].split(',')]

    return params


def _correct_type(param, values):

    vs = values.split(';')
    values = []
    for v in vs:
        if param == 'feature_maps' or param == 'levels' or param == 'epochs':
            v_ = int(v)
        elif param == 'skip_connections' or param == 'residual_connections':
            v_ = bool(int(v))
        elif param == 'dropout' or param == 'lr' or param == 'lambda_mmd' or param == 'lambda_dat' or \
                param == 'lambda_rec' or param == 'lambda_o' or param == 'lambda_w':
            v_ = float(v)
        elif param == 'input_shape':
            v_ = [int(item) for item in v.split(',')]
        else:
            v_ = v
        values.append(v_)

    return param, values


def parse_search_grid(sg_str):

    sg = sg_str.split('#')
    search_grid = {}
    for s in sg:
        param, values = s.split(':')
        param, values = _correct_type(param, values)
        search_grid[param] = values

    return search_grid


def process_seconds(s):
    """
    Processes an amount of seconds to (hours, minutes, seconds)

    :param s: an amount of seconds
    :return: a tuple (h, m, s) that corresponds with the amount of hours, minutes and seconds, respectively
    """

    h = s // 3600
    s -= h*3600
    m = s // 60
    s -= m*60

    return h, m, s