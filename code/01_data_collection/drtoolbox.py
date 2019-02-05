import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from distutils.spawn import find_executable
from glob import glob

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OctaveProjection(BaseEstimator, TransformerMixin):
    def __init__(self, projection, command, verbose):
        self.known_projections = [
            'ProbPCA',
            'GDA',
            'LLC',
            'ManifoldChart',
            'CFA',
            'MVU',
            'FastMVU',
            'CCA',
            'LandmarkMVU',
            'GPLVM',
            'NCA',
            'MCML',
            'LMNN',
            'Sammon']

        self.projection = projection
        self.command = command
        self.verbose = verbose

        if self.projection not in self.known_projections:
            raise ValueError('Invalid projection name: %s. Valid values are %s' % (self.projection, ','.join(self.known_projections)))

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
            raise ValueError('Error looking for projection file %s' % proj_file)

        X_new = np.loadtxt(proj_file, delimiter=',')
        return X_new

    def _run(self, X, y, cmdargs):
        if not find_executable(self.command):
            raise ValueError('Command %s not found' % self.command)

        self._send_data(X, y)

        libpath = os.path.dirname(self.command)

        cmdline = ['octave', '--built-in-docstrings-file', 'built-in-docstrings', '-qf', self.command, libpath,
                   self.tmp_file.name, self.projection] + [str(x) for x in cmdargs]

        if self.verbose:
            print('#################################################')
            print(' '.join(cmdline))

        rc = subprocess.run(cmdline, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=86400, check=True)

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

        try:
            X_new = self._receive_data()
            return X_new
        except:
            print('Error running projection')
            print('return code: ', rc.returncode)
            print('stdout:')
            print('_________________________________________________')
            print(rc.stdout)
            print('_________________________________________________')
            print('stderr:')
            print('_________________________________________________')
            print(rc.stderr)

            reason, _, tb = sys.exc_info()
            print(reason)
            traceback.print_tb(tb)
            raise('Error running projection')


class ProbPCA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', max_iter=200, verbose=False):
        super(ProbPCA, self).__init__(
            projection='ProbPCA', command=command, verbose=verbose)
        self.set_params(command, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', max_iter=200, verbose=False):
        self.command = command
        self.verbose = verbose
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(ProbPCA, self)._run(X, y, [self.max_iter])


class GDA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', kernel='gauss', verbose=False):
        super(GDA, self).__init__(
            projection='GDA', command=command, verbose=verbose)
        self.set_params(command, kernel, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', kernel='gauss', verbose=False):
        self.command = command
        self.verbose = verbose
        self.kernel = kernel

    def fit_transform(self, X, y=None):
        return super(GDA, self)._run(X, y, [self.kernel])


class MCML(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', verbose=False):
        super(MCML, self).__init__(
            projection='MCML', command=command, verbose=verbose)
        self.set_params(command, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', verbose=False):
        self.command = command
        self.verbose = verbose

    def fit_transform(self, X, y=None):
        return super(MCML, self)._run(X, y, [])


class Sammon(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', verbose=False):
        super(Sammon, self).__init__(
            projection='Sammon', command=command, verbose=verbose)
        self.set_params(command, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', verbose=False):
        self.command = command
        self.verbose = verbose

    def fit_transform(self, X, y=None):
        return super(Sammon, self)._run(X, y, [])


class LMNN(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=3, verbose=False):
        super(LMNN, self).__init__(
            projection='LMNN', command=command, verbose=verbose)
        self.set_params(command, k, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=3, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k

    def fit_transform(self, X, y=None):
        return super(LMNN, self)._run(X, y, [self.k])


class MVU(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        super(MVU, self).__init__(
            projection='MVU', command=command, verbose=verbose)
        self.set_params(command, k, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k

    def fit_transform(self, X, y=None):
        return super(MVU, self)._run(X, y, [self.k])


class FastMVU(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        super(FastMVU, self).__init__(
            projection='FastMVU', command=command, verbose=verbose)
        self.set_params(command, k, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k

    def fit_transform(self, X, y=None):
        return super(FastMVU, self)._run(X, y, [self.k])


class LandmarkMVU(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k1=3, k2=12, verbose=False):
        super(LandmarkMVU, self).__init__(
            projection='LandmarkMVU', command=command, verbose=verbose)
        self.set_params(command, k1, k2, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k1=3, k2=12, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k1 = k1
        self.k2 = k2

    def fit_transform(self, X, y=None):
        return super(LandmarkMVU, self)._run(X, y, [self.k1, self.k2])


class CCA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        super(CCA, self).__init__(
            projection='CCA', command=command, verbose=verbose)
        self.set_params(command, k, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k

    def fit_transform(self, X, y=None):
        return super(CCA, self)._run(X, y, [self.k])


class GPLVM(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', sigma=1.0, verbose=False):
        super(GPLVM, self).__init__(
            projection='GPLVM', command=command, verbose=verbose)
        self.set_params(command, sigma, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', sigma=1.0, verbose=False):
        self.command = command
        self.verbose = verbose
        self.sigma = sigma

    def fit_transform(self, X, y=None):
        return super(GPLVM, self)._run(X, y, [self.sigma])


class NCA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', lambd=0.0, verbose=False):
        super(NCA, self).__init__(
            projection='NCA', command=command, verbose=verbose)
        self.set_params(command, lambd, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', lambd=0.0, verbose=False):
        self.command = command
        self.verbose = verbose
        self.lambd = lambd

    def fit_transform(self, X, y=None):
        return super(NCA, self)._run(X, y, [self.lambd])


class LLC(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, n_analyzers=20, max_iter=200, verbose=False):
        super(LLC, self).__init__(
            projection='LLC', command=command, verbose=verbose)
        self.set_params(command, k, n_analyzers, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, n_analyzers=20, max_iter=200, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k
        self.n_analyzers = n_analyzers
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(LLC, self)._run(X, y, [self.k, self.n_analyzers, self.max_iter])


class ManifoldChart(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', n_analyzers=40, max_iter=200, verbose=False):
        super(ManifoldChart, self).__init__(
            projection='ManifoldChart', command=command, verbose=verbose)
        self.set_params(command, n_analyzers, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', n_analyzers=40, max_iter=200, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_analyzers = n_analyzers
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(ManifoldChart, self)._run(X, y, [self.n_analyzers, self.max_iter])


class CFA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', n_analyzers=40, max_iter=200, verbose=False):
        super(CFA, self).__init__(
            projection='CFA', command=command, verbose=verbose)
        self.set_params(command, n_analyzers, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', n_analyzers=40, max_iter=200, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_analyzers = n_analyzers
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(CFA, self)._run(X, y, [self.n_analyzers, self.max_iter])
