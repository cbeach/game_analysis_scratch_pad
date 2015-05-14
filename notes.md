# Currently working on
The algorithm finds runs properly


# Finding Tiles
FFT analysis returns an array of possible tiling frequencies. This data is cleaned in several ways.
1. The mean and standard deviation of the raw FFT data is found. Peaks that are below 2.5 sigma are removed
2. It is assumed that all tiles are the same size (valid for now, probably not later). If this assumtion holds, the FFT data should contain the tile size and several of it's harmonics.
3. Several images with strong peaks in the filtered frequencies should be hashed (more on that later) and analyzed to find the true tile size.
