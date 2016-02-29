# Couchbase 

counters:
    game_number,
    play_number:<game_number>

games: game:<game_number>: {
    'id': <game_number>,
    'name': <string_name>,
    'system': <system>,
    'file_name': <path>,
}

Sprites: sprites:<game> [
    {
        'id': <sprite_id>,
        'path': <absolute_path>'
    }, ...
]

Frame keys: frame:<game_number>:<play_number>:<frame_number> {
    'frame_number': <int>,
    'play_number': <int>,
    'game_number': <int>,
    'path': <absolute_path>,
    'spries': [
        {
            'id': <sprite_id>,
            'bounding_boxes': [
                {
                    'x1': <int>,
                    'y1': <int>,
                    'x2': <int>,
                    'y2': <int>,
                },
            ]
        }
    ],
    'signal': {
        'up': false,
        'down': true,
        ...
    }
}


# Currently working on
The algorithm finds runs properly


# Finding Tiles
FFT analysis returns an array of possible tiling frequencies. This data is cleaned in several ways.
1. The mean and standard deviation of the raw FFT data is found. Peaks that are below 2.5 sigma are removed
2. It is assumed that all tiles are the same size (valid for now, probably not later). If this assumtion holds, the FFT data should contain the tile size and several of it's harmonics.
3. Several images with strong peaks in the filtered frequencies should be hashed (more on that later) and analyzed to find the true tile size.
