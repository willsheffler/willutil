import numpy as np

tetrahedral_frames = np.array([[(+1.000000, +0.000000, +0.000000, +0.000000),
                                (+0.000000, +1.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +1.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+1.000000, +0.000000, +0.000000, +0.000000),
                                (+0.000000, -1.000000, +0.000000, +0.000000),
                                (+0.000000, -0.000000, -1.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, +1.000000, -0.000000, +0.000000),
                                (-0.000000, +0.000000, +1.000000, +0.000000),
                                (+1.000000, -0.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, -1.000000, +0.000000, +0.000000),
                                (-0.000000, -0.000000, -1.000000, +0.000000),
                                (+1.000000, +0.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, +1.000000, -0.000000, +0.000000),
                                (+0.000000, -0.000000, -1.000000, +0.000000),
                                (-1.000000, +0.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.000000, +0.000000, +1.000000, +0.000000),
                                (+1.000000, -0.000000, +0.000000, +0.000000),
                                (+0.000000, +1.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, -1.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +1.000000, +0.000000),
                                (-1.000000, -0.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.000000, -0.000000, -1.000000, +0.000000),
                                (+1.000000, +0.000000, -0.000000, +0.000000),
                                (+0.000000, -1.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, -0.000000, -1.000000, +0.000000),
                                (-1.000000, -0.000000, -0.000000, +0.000000),
                                (-0.000000, +1.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.000000, +0.000000, +1.000000, +0.000000),
                                (-1.000000, +0.000000, -0.000000, +0.000000),
                                (-0.000000, -1.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-1.000000, -0.000000, -0.000000, +0.000000),
                                (-0.000000, +1.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, -1.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-1.000000, +0.000000, -0.000000, +0.000000),
                                (-0.000000, -1.000000, +0.000000, +0.000000),
                                (-0.000000, +0.000000, +1.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)]])

octahedral_frames = np.array([[(+0.000000, +1.000000, +0.000000, +0.000000),
                               (+1.000000, +0.000000, -0.000000, +0.000000),
                               (-0.000000, +0.000000, -1.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, -0.000000, +1.000000, +0.000000),
                               (+1.000000, +0.000000, -0.000000, +0.000000),
                               (-0.000000, +1.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+1.000000, +0.000000, -0.000000, +0.000000),
                               (+0.000000, +1.000000, +0.000000, +0.000000),
                               (+0.000000, -0.000000, +1.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+1.000000, +0.000000, -0.000000, +0.000000),
                               (+0.000000, -0.000000, +1.000000, +0.000000),
                               (+0.000000, -1.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-0.000000, +0.000000, -1.000000, +0.000000),
                               (+0.000000, +1.000000, +0.000000, +0.000000),
                               (+1.000000, -0.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-0.000000, +1.000000, +0.000000, +0.000000),
                               (+0.000000, -0.000000, +1.000000, +0.000000),
                               (+1.000000, +0.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, +1.000000, +0.000000, +0.000000),
                               (-0.000000, +0.000000, -1.000000, +0.000000),
                               (-1.000000, +0.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, -0.000000, +1.000000, +0.000000),
                               (-0.000000, +1.000000, +0.000000, +0.000000),
                               (-1.000000, -0.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, -1.000000, -0.000000, +0.000000),
                               (+1.000000, +0.000000, +0.000000, +0.000000),
                               (+0.000000, -0.000000, +1.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+1.000000, -0.000000, -0.000000, +0.000000),
                               (-0.000000, +0.000000, -1.000000, +0.000000),
                               (+0.000000, +1.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+1.000000, +0.000000, +0.000000, +0.000000),
                               (+0.000000, -1.000000, -0.000000, +0.000000),
                               (-0.000000, +0.000000, -1.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-0.000000, +0.000000, -1.000000, +0.000000),
                               (+1.000000, -0.000000, -0.000000, +0.000000),
                               (-0.000000, -1.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-1.000000, +0.000000, +0.000000, +0.000000),
                               (+0.000000, +1.000000, -0.000000, +0.000000),
                               (-0.000000, +0.000000, -1.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-1.000000, -0.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +1.000000, +0.000000),
                               (-0.000000, +1.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, -0.000000, +1.000000, +0.000000),
                               (+0.000000, -1.000000, -0.000000, +0.000000),
                               (+1.000000, +0.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, +1.000000, -0.000000, +0.000000),
                               (-1.000000, +0.000000, +0.000000, +0.000000),
                               (+0.000000, -0.000000, +1.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, +0.000000, +1.000000, +0.000000),
                               (-1.000000, -0.000000, +0.000000, +0.000000),
                               (+0.000000, -1.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, -1.000000, -0.000000, +0.000000),
                               (-0.000000, -0.000000, +1.000000, +0.000000),
                               (-1.000000, -0.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-0.000000, -1.000000, -0.000000, +0.000000),
                               (-0.000000, +0.000000, -1.000000, +0.000000),
                               (+1.000000, -0.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-0.000000, +0.000000, -1.000000, +0.000000),
                               (-1.000000, +0.000000, +0.000000, +0.000000),
                               (+0.000000, +1.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, +0.000000, -1.000000, +0.000000),
                               (-0.000000, -1.000000, -0.000000, +0.000000),
                               (-1.000000, +0.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-1.000000, +0.000000, +0.000000, +0.000000),
                               (-0.000000, +0.000000, -1.000000, +0.000000),
                               (-0.000000, -1.000000, +0.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(-1.000000, -0.000000, -0.000000, +0.000000),
                               (+0.000000, -1.000000, -0.000000, +0.000000),
                               (-0.000000, -0.000000, +1.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)],
                              [(+0.000000, -1.000000, -0.000000, +0.000000),
                               (-1.000000, -0.000000, -0.000000, +0.000000),
                               (+0.000000, +0.000000, -1.000000, +0.000000),
                               (+0.000000, +0.000000, +0.000000, +1.000000)]])

icosahedral_frames = np.array([[(+1.000000, +0.000000, +0.000000, +0.000000),
                                (+0.000000, +1.000000, +0.000000, +0.000000),
                                (+0.000000, -0.000000, +1.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.809017, +0.309017, +0.500000, +0.000000),
                                (+0.309017, +0.500000, -0.809017, +0.000000),
                                (-0.500000, +0.809017, +0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+1.000000, +0.000000, +0.000000, +0.000000),
                                (+0.000000, -1.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, -1.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.809017, +0.309017, +0.500000, +0.000000),
                                (-0.309017, -0.500000, +0.809017, +0.000000),
                                (+0.500000, -0.809017, -0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.809017, +0.309017, -0.500000, +0.000000),
                                (+0.309017, +0.500000, +0.809017, +0.000000),
                                (+0.500000, -0.809017, +0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.809017, +0.309017, -0.500000, +0.000000),
                                (-0.309017, -0.500000, -0.809017, +0.000000),
                                (-0.500000, +0.809017, -0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.809017, -0.309017, +0.500000, +0.000000),
                                (+0.309017, -0.500000, -0.809017, +0.000000),
                                (+0.500000, +0.809017, -0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.809017, -0.309017, +0.500000, +0.000000),
                                (-0.309017, +0.500000, +0.809017, +0.000000),
                                (-0.500000, -0.809017, +0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.809017, -0.309017, -0.500000, +0.000000),
                                (+0.309017, -0.500000, +0.809017, +0.000000),
                                (-0.500000, -0.809017, -0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.809017, -0.309017, -0.500000, +0.000000),
                                (-0.309017, +0.500000, -0.809017, +0.000000),
                                (+0.500000, +0.809017, +0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.500000, +0.809017, +0.309017, +0.000000),
                                (+0.809017, -0.309017, -0.500000, +0.000000),
                                (-0.309017, +0.500000, -0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.500000, +0.809017, +0.309017, +0.000000),
                                (-0.809017, +0.309017, +0.500000, +0.000000),
                                (+0.309017, -0.500000, +0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.500000, +0.809017, -0.309017, +0.000000),
                                (+0.809017, -0.309017, +0.500000, +0.000000),
                                (+0.309017, -0.500000, -0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.500000, +0.809017, -0.309017, +0.000000),
                                (-0.809017, +0.309017, -0.500000, +0.000000),
                                (-0.309017, +0.500000, +0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.500000, -0.809017, +0.309017, +0.000000),
                                (+0.809017, +0.309017, -0.500000, +0.000000),
                                (+0.309017, +0.500000, +0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.500000, -0.809017, +0.309017, +0.000000),
                                (-0.809017, -0.309017, +0.500000, +0.000000),
                                (-0.309017, -0.500000, -0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.500000, -0.809017, -0.309017, +0.000000),
                                (+0.809017, +0.309017, +0.500000, +0.000000),
                                (-0.309017, -0.500000, +0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.500000, -0.809017, -0.309017, +0.000000),
                                (-0.809017, -0.309017, -0.500000, +0.000000),
                                (+0.309017, +0.500000, -0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.309017, +0.500000, +0.809017, +0.000000),
                                (+0.500000, -0.809017, +0.309017, +0.000000),
                                (+0.809017, +0.309017, -0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.309017, +0.500000, +0.809017, +0.000000),
                                (-0.500000, +0.809017, -0.309017, +0.000000),
                                (-0.809017, -0.309017, +0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.309017, +0.500000, -0.809017, +0.000000),
                                (+0.500000, -0.809017, -0.309017, +0.000000),
                                (-0.809017, -0.309017, -0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.309017, +0.500000, -0.809017, +0.000000),
                                (-0.500000, +0.809017, +0.309017, +0.000000),
                                (+0.809017, +0.309017, +0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.309017, -0.500000, +0.809017, +0.000000),
                                (+0.500000, +0.809017, +0.309017, +0.000000),
                                (-0.809017, +0.309017, +0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.309017, -0.500000, +0.809017, +0.000000),
                                (-0.500000, -0.809017, -0.309017, +0.000000),
                                (+0.809017, -0.309017, -0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.309017, -0.500000, -0.809017, +0.000000),
                                (+0.500000, +0.809017, -0.309017, +0.000000),
                                (+0.809017, -0.309017, +0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.309017, -0.500000, -0.809017, +0.000000),
                                (-0.500000, -0.809017, +0.309017, +0.000000),
                                (-0.809017, +0.309017, -0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, +1.000000, +0.000000, +0.000000),
                                (+0.000000, -0.000000, +1.000000, +0.000000),
                                (+1.000000, -0.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, +1.000000, +0.000000, +0.000000),
                                (-0.000000, +0.000000, -1.000000, +0.000000),
                                (-1.000000, +0.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, +0.000000, +1.000000, +0.000000),
                                (+1.000000, -0.000000, -0.000000, +0.000000),
                                (+0.000000, +1.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, +0.000000, +1.000000, +0.000000),
                                (-1.000000, +0.000000, +0.000000, +0.000000),
                                (-0.000000, -1.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, -1.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +1.000000, +0.000000),
                                (-1.000000, -0.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, -1.000000, +0.000000, +0.000000),
                                (-0.000000, -0.000000, -1.000000, +0.000000),
                                (+1.000000, +0.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, -0.000000, -1.000000, +0.000000),
                                (+1.000000, +0.000000, +0.000000, +0.000000),
                                (+0.000000, -1.000000, +0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(+0.000000, -0.000000, -1.000000, +0.000000),
                                (-1.000000, -0.000000, -0.000000, +0.000000),
                                (-0.000000, +1.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-1.000000, -0.000000, +0.000000, +0.000000),
                                (+0.000000, -1.000000, -0.000000, +0.000000),
                                (+0.000000, +0.000000, +1.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-1.000000, -0.000000, +0.000000, +0.000000),
                                (-0.000000, +1.000000, +0.000000, +0.000000),
                                (-0.000000, -0.000000, -1.000000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.809017, +0.309017, +0.500000, +0.000000),
                                (+0.309017, -0.500000, +0.809017, +0.000000),
                                (+0.500000, +0.809017, +0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.809017, +0.309017, +0.500000, +0.000000),
                                (-0.309017, +0.500000, -0.809017, +0.000000),
                                (-0.500000, -0.809017, -0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.809017, +0.309017, -0.500000, +0.000000),
                                (+0.309017, -0.500000, -0.809017, +0.000000),
                                (-0.500000, -0.809017, +0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.809017, +0.309017, -0.500000, +0.000000),
                                (-0.309017, +0.500000, +0.809017, +0.000000),
                                (+0.500000, +0.809017, -0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.809017, -0.309017, +0.500000, +0.000000),
                                (+0.309017, +0.500000, +0.809017, +0.000000),
                                (-0.500000, +0.809017, -0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.809017, -0.309017, +0.500000, +0.000000),
                                (-0.309017, -0.500000, -0.809017, +0.000000),
                                (+0.500000, -0.809017, +0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.809017, -0.309017, -0.500000, +0.000000),
                                (+0.309017, +0.500000, -0.809017, +0.000000),
                                (+0.500000, -0.809017, -0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.809017, -0.309017, -0.500000, +0.000000),
                                (-0.309017, -0.500000, +0.809017, +0.000000),
                                (-0.500000, +0.809017, +0.309017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.500000, +0.809017, +0.309017, +0.000000),
                                (+0.809017, +0.309017, +0.500000, +0.000000),
                                (+0.309017, +0.500000, -0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.500000, +0.809017, +0.309017, +0.000000),
                                (-0.809017, -0.309017, -0.500000, +0.000000),
                                (-0.309017, -0.500000, +0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.500000, +0.809017, -0.309017, +0.000000),
                                (+0.809017, +0.309017, -0.500000, +0.000000),
                                (-0.309017, -0.500000, -0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.500000, +0.809017, -0.309017, +0.000000),
                                (-0.809017, -0.309017, +0.500000, +0.000000),
                                (+0.309017, +0.500000, +0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.500000, -0.809017, +0.309017, +0.000000),
                                (+0.809017, -0.309017, +0.500000, +0.000000),
                                (-0.309017, +0.500000, +0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.500000, -0.809017, +0.309017, +0.000000),
                                (-0.809017, +0.309017, -0.500000, +0.000000),
                                (+0.309017, -0.500000, -0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.500000, -0.809017, -0.309017, +0.000000),
                                (+0.809017, -0.309017, -0.500000, +0.000000),
                                (+0.309017, -0.500000, +0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.500000, -0.809017, -0.309017, +0.000000),
                                (-0.809017, +0.309017, +0.500000, +0.000000),
                                (-0.309017, +0.500000, -0.809017, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.309017, +0.500000, +0.809017, +0.000000),
                                (+0.500000, +0.809017, -0.309017, +0.000000),
                                (-0.809017, +0.309017, -0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.309017, +0.500000, +0.809017, +0.000000),
                                (-0.500000, -0.809017, +0.309017, +0.000000),
                                (+0.809017, -0.309017, +0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.309017, +0.500000, -0.809017, +0.000000),
                                (+0.500000, +0.809017, +0.309017, +0.000000),
                                (+0.809017, -0.309017, -0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.309017, +0.500000, -0.809017, +0.000000),
                                (-0.500000, -0.809017, -0.309017, +0.000000),
                                (-0.809017, +0.309017, +0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.309017, -0.500000, +0.809017, +0.000000),
                                (+0.500000, -0.809017, -0.309017, +0.000000),
                                (+0.809017, +0.309017, +0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.309017, -0.500000, +0.809017, +0.000000),
                                (-0.500000, +0.809017, +0.309017, +0.000000),
                                (-0.809017, -0.309017, -0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.309017, -0.500000, -0.809017, +0.000000),
                                (+0.500000, -0.809017, +0.309017, +0.000000),
                                (-0.809017, -0.309017, +0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)],
                               [(-0.309017, -0.500000, -0.809017, +0.000000),
                                (-0.500000, +0.809017, -0.309017, +0.000000),
                                (+0.809017, +0.309017, -0.500000, +0.000000),
                                (+0.000000, +0.000000, +0.000000, +1.000000)]])