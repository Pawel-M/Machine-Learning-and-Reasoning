import datetime
import os

import tensorflow.keras as kr


def save_model(model, folder, name, add_timestamp=True):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = name
    if add_timestamp:
        now = datetime.datetime.now()
        timestamp = str(now).replace(':', '-')
        file_name += f'_{timestamp}'

    file_name += '.h5'

    path = os.path.join(folder, file_name)
    model.save(path, overwrite=True, include_optimizer=False, save_format='h5')


def load_model(path):
    return kr.models.load_model(path)
