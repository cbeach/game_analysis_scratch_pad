from glob import glob
import os

from couchbase.bucket import Bucket
from couchbase.views.iterator import View
from couchbase.exceptions import KeyExistsError, NotFoundError


def delete_frame_data(bucket):
    view = View(bucket, 'frame_keys', 'frame_keys')
    counter = 0
    for result in view:
        bucket.delete(result.value)
        counter += 1
    print('{} documents deleted'.format(counter))


def initialize_game_rules_bucket(bucket=None):
    if bucket is None:
        bucket = Bucket('couchbase://localhost/game_rules')
    # Create the game document for super mario bro's
    smb = {
        'file_name': '/home/mcsmash/dev/nestopia/smb.nes',
        'system': 'NES',
        'name': 'Super Mario Brothers and Duck Hunt',
    }
    try:
        bucket.insert('game:1', smb)
    except KeyExistsError:
        pass

    sprite_list = []
    for i, fn in enumerate(glob('/home/mcsmash/dev/data/game_playing/sprites/*')):
        extensionless = os.path.splitext(os.path.basename(fn))
        sprite_list.append({
            'id': extensionless,
            'path': os.path.abspath(fn)
        })

    try:
        bucket.insert('sprites:1', sprite_list)
    except KeyExistsError:
        pass

    try:
        bucket.remove('game_number')
    except NotFoundError:
        pass

    try:
        bucket.remove('play_number:1')
    except NotFoundError:
        pass

    bucket.counter('game_number', initial=2)
    bucket.counter('play_number:1', initial=1)


def reset_game_rules_bucket():
    bucket = Bucket('couchbase://localhost/game_rules')
    print('flushing game_rules')
    bucket.flush()
    print('game_rules flushed')
    initialize_game_rules_bucket(bucket)


if __name__ == '__main__':
    initialize_game_rules_bucket()
