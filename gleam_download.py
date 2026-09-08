"""Download raw GLEAM netCDF files from the UGent SFTP server.

The server lays the data out as ``data/<version>/<resolution>/.../<file>.nc``,
where daily files sit under a year directory and monthly/yearly files sit under
a variable directory. In every case the variable is encoded in the filename, so
it is parsed from there and the local tree is rebuilt as
``<download>/<version>/raw/<resolution>/<variable>/<file>.nc`` regardless of
remote layout, with the version written as ``v_4_3_a`` rather than ``v4.3a``.

Which files are fetched is driven by the config: ``version``, the list of
``temporal_resolutions``, and ``variables`` (a list, or ``all``). Files are
transferred in parallel by ``n_processes`` worker processes, each with its own
SFTP connection and its own log file.

Downloads are restartable: a file whose local copy already matches the remote
size is skipped, transfers land on a ``.part`` file that is only renamed into
place once complete, and stray ``.part`` files are swept before and after a run.
"""

import logging
import argparse
import os
import re
import stat
import time
import posixpath
import multiprocessing
from multiprocessing import Pool

from utils.path_utils import (
    load_config,
)
from utils.log_utils import (
    setup_logging,
)

import paramiko

# public GLEAM v4 credentials, as published with the dataset
HOST = 'aether.ugent.be'
PORT = 2225
USERNAME = 'gleamuser'
PASSWORD = 'GLEAM4#h-cel_111'

# top level directory on the server holding every version
REMOTE_ROOT = 'data'
# 'Ep_rad_1980_GLEAM_v4.3a.nc' -> 'Ep_rad'; greedy so multi-part names survive
FILENAME_RE = re.compile(r'^(?P<variable>.+)_\d{4}_GLEAM_')
# 'v4.3a' -> ('4', '3', 'a'), the pieces of the local directory name
VERSION_RE = re.compile(r'^v(?P<major>\d+)\.(?P<minor>\d+)(?P<letter>[a-z])$')
# processName keeps the workers apart in the shared console stream
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(processName)s %(name)s: %(message)s'

LOG = logging.getLogger(__name__)

# per worker SFTP connection, filled in by init_worker; connections cannot be
# shared between processes, so each worker keeps its own here
_WORKER_STATE = {}


def connect():
    """Open an SFTP connection to the GLEAM server.

    Returns:
        tuple: The paramiko (Transport, SFTPClient); close both when done.
    """
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def walk_sftp(sftp, root):
    """Recursively yield (path, SFTPAttributes) for every entry under root.

    Args:
        sftp (paramiko.SFTPClient): An open SFTP connection.
        root (str): Remote directory to walk.

    Yields:
        tuple: The remote path and its attributes, directories included.
    """
    for entry in sftp.listdir_attr(root):
        full_path = posixpath.join(root, entry.filename)
        yield full_path, entry
        if stat.S_ISDIR(entry.st_mode):
            yield from walk_sftp(sftp, full_path)


def parse_variable(filename):
    """Return the GLEAM variable encoded in a filename, or None if unparseable.

    Args:
        filename (str): Basename of a remote file, e.g. 'Ep_1980_GLEAM_v4.3a.nc'.

    Returns:
        str | None: The variable name, or None if the filename does not match.
    """
    match = FILENAME_RE.match(filename)
    return match.group('variable') if match else None


def format_version(version):
    """Rewrite a GLEAM version for use as a directory name.

    Args:
        version (str): Version as written in the config, e.g. 'v4.3a'.

    Returns:
        str: The version with '.' dropped and the parts underscore separated,
            e.g. 'v_4_3_a'.

    Raises:
        ValueError: If the version is not of the form 'v<major>.<minor><letter>'.
    """
    match = VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"cannot parse version {version!r}, expected e.g. 'v4.3a'")
    return 'v_{major}_{minor}_{letter}'.format(**match.groupdict())


def download_root(settings):
    """Root of the local tree for the configured version.

    Args:
        settings (dict): The loaded configuration.

    Returns:
        str: e.g. '<download>/v_4_3_a/raw'.
    """
    return os.path.join(
        settings['directories']['download'], format_version(settings['version']), 'raw'
    )


def list_files(sftp, settings, resolution):
    """List the files to download for one temporal resolution.

    Args:
        sftp (paramiko.SFTPClient): An open SFTP connection.
        settings (dict): The loaded configuration.
        resolution (str): One of 'daily', 'monthly', 'yearly'.

    Returns:
        list: (remote_path, local_path, size) triples, filtered by variable.
    """
    remote_dir = posixpath.join(REMOTE_ROOT, settings['version'], resolution)
    variables = settings['variables']
    # 'variables: all' takes everything, otherwise it is a list of variables
    download_all = isinstance(variables, str) and variables == 'all'

    files = []
    for path, entry in walk_sftp(sftp, remote_dir):
        if stat.S_ISDIR(entry.st_mode) or not entry.filename.endswith('.nc'):
            continue
        variable = parse_variable(entry.filename)
        if variable is None:
            LOG.warning(f'skipping {path}: cannot parse variable from filename')
            continue
        if not download_all and variable not in variables:
            continue
        # flat local layout: the remote year directories are dropped
        local_path = os.path.join(
            download_root(settings), resolution, variable, entry.filename
        )
        # st_size is kept so downloads can be skipped and verified later
        files.append((path, local_path, entry.st_size))
    return files


def format_size(n_bytes):
    """Human readable file size.

    Args:
        n_bytes (int): Size in bytes.

    Returns:
        str: The size in MB, or GB once it passes 1024 MB.
    """
    mb = n_bytes / 1024**2
    return f'{mb:.1f} MB' if mb < 1024 else f'{mb / 1024:.2f} GB'


def worker_log_file(settings, worker=None):
    """Path of the per-process log file, for the current worker unless one is named.

    Args:
        settings (dict): The loaded configuration.
        worker (str, optional): Worker name to build a path for. Defaults to the
            name of the calling process.

    Returns:
        str: e.g. '<logs>/gleam_download_forkserverpoolworker-1.log'.
    """
    stem, extension = os.path.splitext(settings['log_file'])
    worker = worker or multiprocessing.current_process().name.lower()
    return os.path.join(settings['directories']['logs'], f'{stem}_{worker}{extension}')


def init_worker(settings):
    """Give each worker process its own log file and SFTP connection.

    Runs once per worker at pool startup, so a connection is opened once and
    reused for every file that worker handles.

    Args:
        settings (dict): The loaded configuration.
    """
    setup_logging(worker_log_file(settings), fmt=LOG_FORMAT)
    transport, sftp = connect()
    _WORKER_STATE['transport'] = transport
    _WORKER_STATE['sftp'] = sftp
    LOG.info(f'worker ready, connected to {HOST}:{PORT}')


def download_file(job):
    """Download a single file, skipping it if a complete local copy already exists.

    Args:
        job (tuple): (index, total, remote_path, local_path, size); index and
            total are only used to tag the log lines.

    Returns:
        bool: True if the file is in place afterwards, False if the transfer failed.
    """
    index, total, remote_path, local_path, size = job
    tag = f'[{index}/{total}]'

    # name and size both match the server, so this file is already done
    if os.path.exists(local_path) and os.path.getsize(local_path) == size:
        LOG.info(f'{tag} skipping {remote_path}: already downloaded')
        return True

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    # transfer to a temporary name so an interrupted run never leaves a file
    # that looks complete to the skip check above
    tmp_path = local_path + '.part'
    LOG.info(f'{tag} starting {remote_path} ({format_size(size)}) -> {local_path}')
    start = time.monotonic()
    try:
        _WORKER_STATE['sftp'].get(remote_path, tmp_path)
        os.replace(tmp_path, local_path)
    except Exception as error:
        # keep going with the other files; failures are reported at the end
        LOG.error(f'{tag} failed to download {remote_path}: {error}')
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    elapsed = time.monotonic() - start
    rate = size / 1024**2 / elapsed if elapsed else 0
    LOG.info(
        f'{tag} finished {remote_path} ({format_size(size)}) '
        f'in {elapsed:.1f} s ({rate:.1f} MB/s)'
    )
    return True


def cleanup_partial_files(download_dir):
    """Remove leftover *.part files from a run that was killed mid-transfer.

    Note this removes every *.part under the directory, so two runs must not
    share a download directory.

    Args:
        download_dir (str): Root of the local download tree.

    Returns:
        int: Number of files removed.
    """
    removed = 0
    for root, _, filenames in os.walk(download_dir):
        for filename in filenames:
            if not filename.endswith('.part'):
                continue
            path = os.path.join(root, filename)
            try:
                os.remove(path)
            except OSError as error:
                LOG.error(f'could not remove partial download {path}: {error}')
                continue
            LOG.warning(f'removed partial download {path}')
            removed += 1
    LOG.info(f'cleaned up {removed} partial downloads in {download_dir}')
    return removed


def verify_downloads(jobs):
    """Check every expected file is on disk with the remote size, and log the results.

    Args:
        jobs (list): The (index, total, remote_path, local_path, size) tuples
            that were handed to the pool.

    Returns:
        bool: True if every file is present and complete.
    """
    LOG.info(f'verifying {len(jobs)} downloaded files')
    missing = []
    incomplete = []
    verified_bytes = 0
    for _, _, remote_path, local_path, size in jobs:
        if not os.path.exists(local_path):
            missing.append((remote_path, local_path))
            continue
        local_size = os.path.getsize(local_path)
        if local_size != size:
            incomplete.append((remote_path, local_path, local_size, size))
            continue
        verified_bytes += size

    # list the problem files first, then the summary line
    for remote_path, local_path in missing:
        LOG.error(f'missing: {local_path} (remote {remote_path})')
    for remote_path, local_path, local_size, size in incomplete:
        LOG.error(
            f'size mismatch: {local_path} is {format_size(local_size)}, '
            f'remote {remote_path} is {format_size(size)}'
        )

    n_ok = len(jobs) - len(missing) - len(incomplete)
    LOG.info(
        f'verified {n_ok}/{len(jobs)} files ({format_size(verified_bytes)}), '
        f'{len(missing)} missing, {len(incomplete)} incomplete'
    )
    return not missing and not incomplete


def main(settings):

    os.makedirs(download_root(settings), exist_ok=True)
    os.makedirs(settings['directories']['logs'], exist_ok=True)
    log_file = os.path.join(settings['directories']['logs'], settings['log_file'])
    setup_logging(log_file, fmt=LOG_FORMAT)

    LOG.info(
        f'downloading GLEAM {settings["version"]} data '
        f'({settings["temporal_resolutions"]}, variables: {settings["variables"]}) from {HOST}'
    )

    # build the full job list up front, over a single connection, so the total
    # volume is known before any transfer starts
    transport, sftp = connect()
    try:
        jobs = []
        for resolution in settings['temporal_resolutions']:
            resolution_jobs = list_files(sftp, settings, resolution)
            LOG.info(f'found {len(resolution_jobs)} {resolution} files')
            jobs.extend(resolution_jobs)
    finally:
        sftp.close()
        transport.close()

    total_bytes = sum(size for _, _, size in jobs)
    LOG.info(f'found {len(jobs)} files in total ({format_size(total_bytes)})')

    # number the jobs so each log line says which file of how many it is
    jobs = [(index, len(jobs), *job) for index, job in enumerate(jobs, start=1)]

    # clear debris from any previous run before the workers start writing
    cleanup_partial_files(download_root(settings))

    n_processes = min(settings['n_processes'], len(jobs)) or 1
    LOG.info(
        f'starting {n_processes} worker processes, '
        f'each logging to {worker_log_file(settings, worker="<worker>")}'
    )
    try:
        # chunksize=1 hands out one file at a time, so a worker that draws a
        # large file does not hold up a whole pre-assigned block
        with Pool(processes=n_processes, initializer=init_worker, initargs=(settings,)) as pool:
            results = pool.map(download_file, jobs, chunksize=1)
    except BaseException as error:
        # covers ctrl-c too; the pool has terminated and joined its workers by
        # the time this runs, so nothing is still writing to a .part file
        LOG.error(f'download interrupted ({type(error).__name__}), cleaning up before exiting')
        cleanup_partial_files(download_root(settings))
        raise
    cleanup_partial_files(download_root(settings))

    for (index, _, remote_path, _, _), ok in zip(jobs, results):
        if not ok:
            LOG.error(f'[{index}/{len(jobs)}] did not download: {remote_path}')
    LOG.info(f'downloaded {results.count(True)} files, {results.count(False)} failed')

    # final check against the remote sizes, in the main log; a rerun picks up
    # exactly the files reported here
    if not verify_downloads(jobs):
        LOG.error('download incomplete, rerun to retry the files listed above')
        raise SystemExit(1)

    LOG.info('all files downloaded and verified')
    LOG.info('done :-)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='download raw gleam netcdf files over sftp.'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file.',
    )
    args = parser.parse_args()
    settings = load_config(args.config)
    main(settings)
