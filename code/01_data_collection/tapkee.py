import os
import numpy as np
from glob import glob
import tempfile
import subprocess
from distutils.spawn import find_executable
from sklearn.base import BaseEstimator, TransformerMixin


class TapkeeCLIProjection(BaseEstimator, TransformerMixin):
    def __init__(self, projection, command, verbose):
        self.known_projections = [
            'diffusion_map',
            'manifold_sculpting',
            'stochastic_proximity_embedding',
            'locality_preserving_projections',
            'linear_local_tangent_space_alignment',
            'neighborhood_preserving_embedding',
            'landmark_multidimensional_scaling']

        self.projection = projection
        self.command = command
        self.verbose = verbose

        if self.projection not in self.known_projections:
            raise ValueError('Invalid projection name: %s. Valid values are %s'
                             % (self.projection, ','.join(self.known_projections)))

    def fit_transform(self, X, y=None):
        raise Exception('Not implemented')

    def _send_data(self, X, y):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_file = tempfile.NamedTemporaryFile(
            mode='w', dir=self.tmp_dir.name, suffix='.data', delete=False)

        np.savetxt(self.tmp_file.name, X, delimiter=',')

        return self.tmp_dir.name

    def _receive_data(self):
        proj_file = self.tmp_file.name + '.prj'

        if not os.path.exists(proj_file):
            raise ValueError(
                'Error looking for projection file %s' % proj_file)

        X_new = np.loadtxt(proj_file, delimiter=',')

        return X_new

    def _run(self, X, y, cmdargs):
        if not find_executable(self.command):
            raise ValueError('Command %s not found' % self.command)

        self._send_data(X, y)

        cmdline = [self.command, '--method', self.projection, '--eigen-method', 'dense', '--neighbors-method', 'brute',
                   '-i', self.tmp_file.name, '-o', self.tmp_file.name + '.prj', ] + [str(x) for x in cmdargs]

        if self.verbose:
            cmdline.append('--verbose')
            print('#################################################')
            print(' '.join(cmdline))

        rc = subprocess.run(cmdline, universal_newlines=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=86400, check=True)

        if self.verbose:
            print('return code: ', rc.returncode)
            print('stdout:')
            print('_________________________________________________')
            print(rc.stdout)
            print('_________________________________________________')
            print('stderr:')
            print('_________________________________________________')
            print(rc.stderr)
            print('#################################################')

        return self._receive_data()


class DiffusionMaps(TapkeeCLIProjection):
    def __init__(self, command=os.getcwd() + '/tapkee/tapkee', t=2, width=1.0, verbose=False):
        super(DiffusionMaps, self).__init__(
            projection='diffusion_map', command=command, verbose=verbose)
        self.set_params(command, t, width, verbose)

    def set_params(self, command=os.getcwd() + '/tapkee/tapkee', t=2, width=1.0, verbose=False):
        self.command = command
        self.verbose = verbose
        self.t = t
        self.width = width

    def fit_transform(self, X, y=None):
        return super(DiffusionMaps, self)._run(X, y,
                                               ['--timesteps', self.t,
                                                '--gaussian-width', self.width])


# Not working: hangs during execution most of the time
# class ManifoldSculpting(TapkeeCLIProjection):
#     def __init__(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, squishing_rate=0.8, max_iter=80, verbose=False):
#         super(ManifoldSculpting, self).__init__(
#             projection='manifold_sculpting', command=command, verbose=verbose)
#         self.set_params(command, n_neighbors,
#                         squishing_rate, max_iter, verbose)

#     def set_params(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, squishing_rate=0.8, max_iter=80, verbose=False):
#         self.command = command
#         self.verbose = verbose
#         self.n_neighbors = n_neighbors
#         self.squishing_rate = squishing_rate
#         self.max_iter = max_iter

#     def fit_transform(self, X, y=None):
#         return super(ManifoldSculpting, self)._run(X, y,
#                                                    ['-k', self.n_neighbors,
#                                                     '--squishing-rate', self.squishing_rate,
#                                                     '--max-iters', self.max_iter])


class StochasticProximityEmbedding(TapkeeCLIProjection):
    def __init__(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=12, n_updates=20, max_iter=0, verbose=False):
        super(StochasticProximityEmbedding, self).__init__(
            projection='stochastic_proximity_embedding', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, n_updates, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=12, n_updates=20, max_iter=0, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.n_updates = n_updates
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(StochasticProximityEmbedding, self)._run(X, y,
                                                              ['-k', self.n_neighbors,
                                                               '--spe-num-updates', self.n_updates,
                                                               '--max-iters', self.max_iter])


class LocalityPreservingProjections(TapkeeCLIProjection):
    def __init__(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, verbose=False):
        super(LocalityPreservingProjections, self).__init__(
            projection='locality_preserving_projections', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, verbose)

    def set_params(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        return super(LocalityPreservingProjections, self)._run(X, y,
                                                               ['-k', self.n_neighbors])


class LinearLocalTangentSpaceAlignment(TapkeeCLIProjection):
    def __init__(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, verbose=False):
        super(LinearLocalTangentSpaceAlignment, self).__init__(
            projection='linear_local_tangent_space_alignment', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, verbose)

    def set_params(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        return super(LinearLocalTangentSpaceAlignment, self)._run(X, y,
                                                                  ['-k', self.n_neighbors])


class NeighborhoodPreservingEmbedding(TapkeeCLIProjection):
    def __init__(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, verbose=False):
        super(NeighborhoodPreservingEmbedding, self).__init__(
            projection='neighborhood_preserving_embedding', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, verbose)

    def set_params(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        return super(NeighborhoodPreservingEmbedding, self)._run(X, y,
                                                                 ['-k', self.n_neighbors])


class LandmarkMDS(TapkeeCLIProjection):
    def __init__(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, verbose=False):
        super(LandmarkMDS, self).__init__(
            projection='landmark_multidimensional_scaling', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, verbose)

    def set_params(self, command=os.getcwd() + '/tapkee/tapkee', n_neighbors=10, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        return super(LandmarkMDS, self)._run(X, y,
                                             ['-k', self.n_neighbors])
