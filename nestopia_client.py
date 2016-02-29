#import curses
import inspect
import json
import os
import random
import socket
import sys
import time

#from matplotlib import pyplot as plt
#from termcolor import cprint
from couchbase.bucket import Bucket
from couchbase.exceptions import NotFoundError
import cv2
import numpy as np

MB = 2 ** 20
port = 9090
DATA_DIR = '/home/mcsmash/dev/data/game_playing/frames/'

buttons = ['up', 'down', 'left', 'right', 'select', 'start', 'a', 'b']


def write_frame_to_couchbase(bucket, game_number, play_number, frame_number, frame_path, signal):
    key = 'frame:{}:{}:{}'.format(game_number, play_number, frame_number)
    return bucket.insert(key, {
        'game_number': game_number,
        'play_number': play_number,
        'frame_number': frame_number,
        'path': frame_path,
        'signal': encode_input(**signal),
        'sprites': [],
    })


def random_control_sequence(start_frame, max_frame_ahead=50, duration_mu_sigma=(100, 25)):
    frame = random.randint(0, max_frame_ahead) + start_frame
    duration = abs(int(random.gauss(*duration_mu_sigma)))
    button_states = [True if random.randint(0, 1) == 1 else False for i in buttons]
    signal = {k: button_states[i] for i, k in enumerate(buttons)}
    return (frame, duration, signal)


def flip_coin():
    return random.randint(0, 1) == 0


def mario_filter(client, start_scree=None):
    """
    frame_triggers: (frame, time_between_checks, signal)
    """
    def func(start_frame, max_frame_ahead=50, duration_mu_sigma=(100, 25)):
        frame, duration, signal = random_control_sequence(start_frame, max_frame_ahead,
                                                          duration_mu_sigma)
        signal['start'] = False
        signal['select'] = False
        signal['up'] = False

        if signal['left'] and signal['right']:
            if flip_coin() is True:
                signal['left'] = False
            else:
                signal['right'] = False

        if signal['left'] is True and random.randint(0, 100) < 75:
            signal['left'] = False
            signal['right'] = True

        return (frame, duration, signal)
    return func


def show_image(image):
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)
    #cv2.destroyAllWindows()


def encode_input(player=0, up=False, down=False, left=False, right=False, select=False,
                 start=False, a=False, b=False, turbo_a=False, turbo_b=False, altspeed=False,
                 insertcoin1=False, insertcoin2=False, fdsflip=False, fdsswitch=False,
                 qsave1=False, qsave2=False, qload1=False, qload2=False, screenshot=False,
                 reset=False, rwstart=False, rwstop=False, fullscreen=False, video_filter=False,
                 scalefactor=False, quit=False):
    buttons = ['up', 'down', 'left', 'right', 'select', 'start', 'a', 'b', 'turbo_a', 'turbo_b',
               'altspeed', 'insertcoin1', 'insertcoin2', 'fdsflip', 'fdsswitch', 'qsave1',
               'qsave2', 'qload1', 'qload2', 'screenshot', 'reset', 'rwstart', 'rwstop',
               'fullscreen', 'video_filter', 'scalefactor', 'quit']

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    pressed_buttons = []
    for arg in args:
        if values[arg] is True and arg in buttons:
            pressed_buttons.append(arg)
    return json.dumps({
        'controls': pressed_buttons,
        'player': player,
    })


class NestopiaClient:
    def __init__(self, rom_file, sequence_source=None, initialization_signals=None):
        self.rom_path = rom_file
        self.frame_number = 0

        # Connect to game server
        self.sock = socket.socket()
        self.host = socket.gethostname()
        self.sock.connect((self.host, port))
        self.sock.send(json.dumps({
            'rom_file': rom_file
        }))

        # Load ROM
        response = json.loads(self.sock.recv(MB))

        # Get video resolution
        width = int(response['width'])
        height = int(response['height'])
        scale = int(response['scale'])
        self.scale = scale
        self.width = width * scale
        self.height = height * scale
        self.depth = 4
        self.frame = np.empty((self.height, self.width, self.depth), dtype=np.uint8)
        self.frame_size = self.height * self.width * self.depth

        self._start_time = time.time()

        # Initalize automated play
        self._game_initializing = initialization_signals is not None
        self._init_sequence = initialization_signals
        self._sequence_source = sequence_source
        self._current_control_sequence = None
        self._sequence_countdown = 0

    def next_frame(self, signal=None):
        signal = self._send_signal(signal)
        self.sock.recv_into(self.frame, self.frame_size, socket.MSG_WAITALL)
        self.frame_number += 1
        return self.frame_number, self.frame[:, :, :3], signal

    def _send_signal(self, signal=None):
        if signal is None:
            signal = self.current_signal
        self.sock.send(encode_input(**signal))
        return signal

    @property
    def sequence_source(self):
        return self._sequence_source

    @sequence_source.setter
    def sequence_source(self, func):
        self._sequence_source = func

    @property
    def current_signal(self):
        if self._game_initializing is True:
            return self._next_game_initialization_signal()

        if self._current_control_sequence is None:
            self._current_control_sequence = self._sequence_source()

        sequence_start = self._current_control_sequence[0]
        sequence_end = self._current_control_sequence[0] + self._current_control_sequence[1]
        if sequence_start <= self.frame_number < sequence_end:
            return self._current_control_sequence[2]
        elif self.frame_number >= sequence_end:
            self._current_control_sequence = self._sequence_source(self.frame_number)
            return self.current_signal
        else:
            return self._current_control_sequence[2]

    def _next_game_initialization_signal(self):
        if self._current_control_sequence is None:
            self._current_control_sequence = self._init_sequence.pop(0)

        if self._current_control_sequence[0] == self.frame_number:
            self._sequence_countdown = self._current_control_sequence[1]

        if self._sequence_countdown > 0:
            self._sequence_countdown -= 1
            return self._current_control_sequence[2]
        else:
            if self.frame_number > self._current_control_sequence[0]:
                try:
                    self._current_control_sequence = self._init_sequence.pop(0)
                    return self._current_control_sequence[2]
                except(IndexError):
                    self._game_initializing = False
                    return self.current_signal
            else:
                return {}


if __name__ == '__main__':
    rom = '/home/mcsmash/dev/nestopia/smb.nes'
    data_base_dir = '/home/mcsmash/dev/data/game_playing/frames'
    bucket = Bucket('couchbase://localhost/game_rules')
    start_screen = '/home/mcsmash/dev/data/game_playing/frames/game_1/trigger_frames/start_screen.png'
    start_screen = cv2.imread(start_screen)

    game_id = 1
    game_info = bucket.get('game:{}'.format(game_id))
    try:
        play_number = bucket.counter('play_number:{}'.format(game_id)).value
    except NotFoundError:
        play_number = bucket.counter('play_number:{}'.format(game_id), initial=1).value

    data_dir = os.path.join(data_base_dir, 'game_{}'.format(game_id))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data_dir = os.path.join(data_dir, 'play_number_{}'.format(play_number))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    init_sequence = [
        (175, 1, {
            'start': True,
        }), (225, 1, {
            'start': True,
        }),
    ]

    client = NestopiaClient(rom, initialization_signals=init_sequence)
    client._sequence_source = mario_filter(client, start_screen)
    while True:
        frame_number, frame, signal = client.next_frame()
        frame_path = os.path.join(data_dir, '{}.png'.format(frame_number))
        show_image(frame[:, :, :3])
        frames_per_second = int(float(client.frame_number) / float(time.time() - client._start_time))
        write_frame_to_couchbase(bucket, game_id, play_number, frame_number, frame_path, signal)
        print('frame: {}, f/s: {}, signal: {}'.format(client.frame_number, frames_per_second, signal))
        cv2.imwrite(frame_path, frame[:, :, :3])
