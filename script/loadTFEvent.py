import os


def load_tf_event(path=r'tb_loggers', event_id : int = None, host_name: str = None):
    assert os.path.exists(path)
    from tensorboard.backend.event_processing import event_accumulator  # import tensorboard event parser
    if os.path.isdir(path):
        if event_id is None or host_name is None:
            path = next(os.walk(path))[-1]
            path.sort()
            path = path[-1]
        else:
            path = os.path.join(path, 'events.out.tfevents.{}.{}'.format(int(event_id), host_name))
    ea = event_accumulator.EventAccumulator(path)  # init EventAccumulator
    ea.Reload()  # load the event
    print('event_keys:', ea.scalars.Keys())  # tensorboard can save Image scalars, but we only observe the scalars.
    return {
        key: {
            'steps': [item.step for item in ea.scalars.Items(key)],
            'values': [item.value for item in ea.scalars.Items(key)]
        } for key in ea.scalars.Keys()
    }
