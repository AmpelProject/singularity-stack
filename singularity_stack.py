#!/usr/bin/env python3

"""
Deploy a Docker application with Singularity, in a manner similar to `docker-stack`
"""

import yaml
import json
import daemon
import daemon.pidfile
import requests

import datetime
import dateutil.parser
import sys, os, stat, shutil
import warnings
import re
import time
import select
import signal
import subprocess
import getpass
import tempfile
import asyncio
import pathlib
import socket
import multiprocessing
import logging

log = logging.getLogger(__name__)

__version__ = "0.4.0"

def start_order(services):
    """
    Yield names of services in the order that they need to be started to
    satisfy the depends_on clauses of each
    """
    def emit_service(name, services):
        for dep in services[name].get('depends_on', []):
            emit_service(dep, services)
        yield name
    for name in services:
        for dep in emit_service(name, services):
            yield dep

def singularity_image(name):
    """
    Return the path for a Singularity image for the given Docker image name.
    If an appropriate image does not exist, pull it
    
    :param name: an image name, of the form organization/name:tag
    """
    basename = name
    if '/' in name:
        _, basename = name.split('/')
    if ':' in basename:
        basename, tag = basename.split(':')
        basename += '-{}'.format(tag)
    
    cwd = os.environ.get('SINGULARITY_CACHEDIR', os.getcwd())
    image_path = os.path.join(cwd, basename+'.simg')
    if not os.path.isfile(image_path):
        subprocess.check_call(['singularity', 'pull', 'docker://{}'.format(name)], cwd=cwd)
    return image_path

overlayfs_is_broken = True

def _bind_secret(name, config):
    if not name in config.get('secrets', dict()):
        raise ValueError("Secret '{}' not defined in configuration".format(name))
    secret = config['secrets'][name]['file']
    perms = os.stat(secret).st_mode
    if (perms & stat.S_IRGRP) or (perms & stat.S_IROTH):
        raise ValueError("Secrets file '{}' should be readable only by its owner".format(secret))
    if overlayfs_is_broken:
        return []
    else:
        return ['-B', '{}:/run/secrets/{}'.format(secret, name)]

def _env_secrets(env, config):
    """
    Overlay mounts may not be available due to EGI-SVG-2018-14213. Find
    idiomatic uses of secrets files and place the contents in the
    environment.
    """
    if overlayfs_is_broken:
        prefix = '/run/secrets/'
        postfix = '_FILE'
        for k in list(env.keys()):
            if k.endswith(postfix) and env[k].startswith(prefix):
                path = env[k]
                name = path[len(prefix):]
                secret = config['secrets'][name]['file']
                del env[k]
                with open(secret) as f:
                    env[k[:-len(postfix)]] = f.read().strip()

def _init_volume(image, source, target):
    """
    Initialize a host directory with the contents of a container path
    """
    pathlib.Path(source).parent.mkdir(parents=True, exist_ok=True)
    tagfile = os.path.join(source, ".singularity-stack-volume")
    if not os.path.exists(tagfile):
        cmd = "cp -r '{}' '{}'".format("/var/singularity/mnt/final/"+target, source)
        subprocess.check_call(["singularity", "mount", image, "sh", "-c", cmd])
        with open(tagfile, 'w') as f:
            f.write('{}:{}'.format(image,target))

def init_volume_cmd(args):
    if os.path.exists(args.dest) and not args.force:
        raise FileExistsError("Destination path {} already exists. Pass --force to overwrite".format(args.dest))
    _init_volume(args.image, args.dest, args.source)

def _parse_volume(volume):
    if isinstance(volume, str):
        if ':' in volume:
            source, dest = volume.split(':')
        else:
            source, dest = volume, volume
        if source.startswith('.') or source.startswith('/') and ':' in volume:
            voltype = 'bind'
        else:
            voltype = 'volume'
        return dict(type=voltype, source=source, target=dest)
    elif isinstance(volume, dict):
        return volume
    else:
        raise TypeError

def singularity_command_line(name, config):
    """
    Start a Singularity instance from the given Docker service definition
    """
    # TODO:
    # entrypoint
    # ports
    # expose
    service = config['services'][name]
    if not 'image' in service:
        raise ValueError("service '{}' is missing an 'image' key".format(name))
    cmd = ['--contain']

    assert '_tempfiles' not in service
    service['_tempfiles'] = list()
    
    # Mount volumes
    for volume in map(_parse_volume, service.get('volumes', [])):
        if volume['type'] == 'volume':
            _init_volume(singularity_image(service['image']), volume['source'], volume['target'])
            cmd += ['-B', '{}:{}'.format(volume['source'], volume.get('target', volume['source']))]
        elif volume['type'] == 'tmpfs':
            tempdir = tempfile.mkdtemp(dir='/dev/shm/')
            service['_tempfiles'].append(tempdir)
            cmd += ['-B', '{}:{}'.format(tempdir, volume['target'])]
        elif volume['type'] == 'bind':
            cmd += ['-B', '{}:{}'.format(volume['source'], volume.get('target', volume['source']))]
        else:
            raise ValueError("Unsupported volume type '{}'".format(volume['type']))
    
    # Fill and bind /etc/hosts file
    with tempfile.NamedTemporaryFile(mode='wt', delete=False) as hosts:
        service['_tempfiles'].append(hosts)
        for entry in service.get('extra_hosts', list()):
            hosts.write('{1}\t{0}\n'.format(*entry.split(':')))
        # Alias the names of all services in the stack to localhost
        # TODO: support placement on other nodes and/or future Singularity
        #       network isolation
        for entry in config['services']:
            hosts.write('{1}\t{0}\n'.format(entry, '127.0.0.1'))
    cmd += ['-B', '{}:/etc/hosts'.format(hosts.name)]
    
    # Bind secrets files
    # NB: unlike Docker, these are not encrypted at rest.
    for secret in service.get('secrets', list()):
        cmd += _bind_secret(secret, config)
    
    cmd.append(singularity_image(service['image']))
    
    return cmd

def _parse_duration(duration):
    """
    Parse a duration in seconds from a docker-compose style duration string
    """
    dt = 0
    units = dict(us=1e-6, s=1, m=60, h=3600)
    for num, unit in re.findall(r'(\d+\.?\d?)([a-z]+)', duration):
        if not unit in units:
            raise ValueError("duration '{}' contains unknown unit '{}'".format(duration, unit))
        dt += float(num)*units[unit]
    if len(duration) > 0 and dt == 0:
        raise ValueError("duration '{}' could not be parsed".format(duration))
    return dt

import hashlib
def _instance_name(*args):
    return hashlib.sha1('.'.join(map(str,args)).encode()).hexdigest()[:8]

def _log_prefix(*args):
    return "/var/tmp/{}.singularity.{}".format(getpass.getuser(), _instance_name(*args))

def _pid_file(*args):
    return '/var/tmp/{}_{}.pid'.format(getpass.getuser(), '_'.join(args))

def _instance_running(name):
    # FIXME: make this check for an exact match
    ret = subprocess.call(['singularity', 'instance.list', name],
        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return ret == 0

def _start_replica_set(app, name, configs):
    service = configs[0]['services'][name]

    # in parent process, wait for service to start listening on exposed ports
    if os.fork() != 0:
        log.debug("Waiting for {}.{} to start".format(app, name))
        for mapping in service.get('ports', []):
            if ':' in mapping:
                src, dest = map(int, mapping.split(':'))
            else:
                src = int(mapping)
                dest = src
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            timeout = 1.
            for _ in range(20):
                try:
                    s.connect((socket.gethostname(), dest))
                    break
                except ConnectionRefusedError:
                    log.debug('connection refused on {}:{}, retry after {:.0f} s'.format(socket.gethostname(), dest, timeout))
                    time.sleep(timeout)
                    timeout *= 1.5
                    continue
            else:
                raise CalledProcessError('{}.{} failed to start'.format(app, name))
            log.info('connected to {}:{}'.format(socket.gethostname(), dest))

        log.info("Started {}.{} (x{})".format(app, name, len(configs)))

        return True
    # in child process, do actual work
    else:
        logfile = open('{}_{}.log'.format(app, name), 'wb')
        log.debug('starting replicas')
        pidfile = daemon.pidfile.PIDLockFile(_pid_file(app, name))
        with daemon.DaemonContext(pidfile=pidfile, stderr=logfile, stdout=logfile, working_directory=os.getcwd()):
            log.debug('entered daemon context')
            _run_replica_set(app, name, configs)
        log.debug('exiting')
        sys.exit(0)

from logging.handlers import RotatingFileHandler
class LogRotator(RotatingFileHandler):
    def format(self, record):
        return json.dumps(record)
    def emit(self, record):
        if self.shouldRollover(record):
            self.doRollover()
        json.dump(record, self.stream)
        self.stream.write('\n')
        self.stream.flush()

class LogEmitter:
    def __init__(self, queue, **kwargs):
        self._queue = queue
        self.extra = kwargs
    def write(self, msg):
        msg = msg.rstrip()
        if len(msg) == 0:
            return
        payload = dict(timestamp=time.time(), msg=msg)
        payload.update(self.extra)
        self._queue.put(payload)

class LogCollector:
    def __init__(self, queue, procs, handler):
        self._queue = queue
        self._procs = procs
        self._handler = handler
    def run(self):
        log.debug('collecting logs')
        for proc in self._procs.values():
            proc.start()
        while len(self._procs) > 0 or not self._queue.empty():
            try:
                msg = self._queue.get()
            except KeyboardInterrupt:
                log.debug('Caught SIGINT; shutting down')
                for proc in self._procs.values():
                    proc.terminate()
                for proc in self._procs.values():
                    proc.join()
                log.debug('Shut down')
                break
            if isinstance(msg, int):
                # got sentinel, reap child process
                self._procs[msg].join()
                del self._procs[msg]
            else:
                self._handler.emit(msg)

def _run_replica_set(app, name, configs):
    replicas = len(configs)
    queue = multiprocessing.Queue()
    procs = {i: multiprocessing.Process(target=_run, args=(app, name, i, configs[i], queue)) for i in range(replicas)}
    handler = LogRotator(_log_prefix(app, name)+".json", maxBytes=2**24, backupCount=16, encoding="utf-8")
    collector = LogCollector(queue, procs, handler)
    #sys.stderr = sys.stdout = LogEmitter(queue, app=app, service=name, source="collector")
    collector.run()

def _run(app, name, replica, config, log_queue):
    try:
        _run_impl(app, name, replica, config, log_queue)
    except:
        import traceback
        traceback.print_exc()
    finally:
        log_queue.put(replica)

def _register(app, name, replica, config):
    """
    Mimic gliderlabs Registrator (TM)
    """
    if not 'x-consul' in config:
        return
    service = config['services'][name]
    env = {k: str(v) for k,v in service.get('environment', {}).items()}
    payload = {
        'name': name,
        'id': '.'.join([app, name, str(replica)]),
        'meta': {}
    }
    prefix = 'SERVICE_'
    for k, v in env.items():
        if not k.startswith(prefix):
            continue
        key = k[len(prefix):]
        if key.lower() in payload:
            payload[key] = v
        elif key.lower() == 'tags':
            payload['tags'] = v.split(',')
        else:
            payload['meta'][key] = v
    for mapping in service.get('ports', []):
        if ':' in mapping:
            src, dest = map(int, mapping.split(':'))
        else:
            src = int(mapping)
            dest = src
        payload['address'] = socket.gethostbyname(socket.gethostname())
        payload['port'] = src
        # check that the service is listening
        payload['check'] = {
            'id': name,
            'tcp': '{}:{}'.format(payload['address'], payload['port']),
            'interval': '60s',
            'timeout': '10s',
        }
        # FIXME: support multiple service ports per container?
        break
    requests.put('{}/v1/agent/service/register'.format(config['x-consul']), json=payload).raise_for_status()

def _deregister(app, name, replica, config):
    if not 'x-consul' in config:
        return
    service_id = '.'.join([app, name, str(replica)])
    requests.put('{}/v1/agent/service/deregister/{}'.format(config['x-consul'], service_id)).raise_for_status()

def _run_impl(app, name, replica, config, log_queue):
    """
    Fork a daemon process that starts the service process in a Singularity
    instance, redirects its stdout/stderr to files, and restarts it on failure
    according to the given `restart_policy`
    """
    sys.stderr = sys.stdout = LogEmitter(log_queue, app=app, service=name, replica=replica, source="nanny")

    service = config['services'][name]
    if 'restart' in service:
        raise ValueError("restart is not supported. use deploy instead")
    instance = _instance_name(app, name, replica)
    # NB: we assume that the default entrypoint simply calls exec on arguments
    # it does not recognize
    cmd = ['singularity', 'run', 'instance://'+instance]
    if 'command' in service:
        assert isinstance(service['command'], list)
        cmd += service['command']
    
    env = {k: str(v) for k,v in service.get('environment', {}).items()}
    if overlayfs_is_broken:
        _env_secrets(env, config)
    
    restart_policy = service.get('deploy', {}).get('restart_policy', {})
    if not restart_policy.get('condition', False) in {'on-failure', 'any', False}:
        raise ValueError("unsupported restart condition '{}'".format(restart_policy['condition']))
    
    max_attempts = 0
    if restart_policy.get('condition','no') in {'on-failure', 'any'}:
        max_attempts = restart_policy.get('max_attempts', sys.maxsize)
    delay = _parse_duration(restart_policy.get('delay', ''))
    
    ret = 0
    for _ in range(-1, max_attempts):
        if not _instance_running(instance):
            print('{} instance is gone!'.format(instance))
            return 1
        stderr = LogEmitter(log_queue, app=app, service=name, replica=replica, source="stderr")
        stdout = LogEmitter(log_queue, app=app, service=name, replica=replica, source="stdout")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _register(app, name, replica, config)
        reads = [proc.stdout.fileno(), proc.stderr.fileno()]
        while True:
            ready = select.select(reads, [], [], 1)[0]
            for fd in ready:
                if fd == reads[0]:
                    line = proc.stdout.readline().decode()
                    stdout.write(line)
                elif fd == reads[1]:
                    line = proc.stderr.readline().decode()
                    stderr.write(line)
            if proc.poll() is not None:
                # process has exited, suck pipes dry
                for fd, sink in ((proc.stdout, stdout), (proc.stderr, stderr)):
                    line = fd.read()
                    if len(line) > 0:
                        sink.write(line.decode())
                break
        ret = proc.wait()
        _deregister(app, name, replica, config)
        if ret == 0 and restart_policy.get('condition', 'no') != 'any':
            break
        else:
            print('sleeping {} before restart'.format(delay))
            time.sleep(delay)
    if ret != 0:
        print('failed permanently with status {}'.format(ret))
    else:
        print('exited cleanly')
    for temp in service.get('_tempfiles', []):
        if hasattr(temp, 'name'):
            temp = getattr(temp, 'name')
        print('removing {}'.format(temp))
        if os.path.isdir(temp):
            shutil.rmtree(temp)
        else:
            os.unlink(temp)
    return ret

import re
def expandvars(path, environ=os.environ):
    """Expand shell variables of form $var and ${var}.  Unknown variables
    raise and error. Adapted from os.posixpath.expandvars"""
    if not isinstance(path, str):
        return path
    if '$' not in path:
        return path
    _varprog = re.compile(r'\$(\w+|\{[^}]*\})', re.ASCII)
    search = _varprog.search
    start = '{'
    end = '}'
    i = 0
    while True:
        m = search(path, i)
        if not m:
            break
        i, j = m.span(0)
        name = m.group(1)
        if name.startswith(start) and name.endswith(end):
            name = name[1:-1]
        try:
            value = environ[name]
        except KeyError:
            i = j
            raise
        else:
            tail = path[j:]
            path = path[:i] + value
            i = len(path)
            path += tail
    return expandvars(path, environ)

def _transform_items(collection, transform):
    if isinstance(collection, list):
        return [_transform_items(c, transform) for c in collection]
    elif isinstance(collection, dict):
        result = {}
        for k,v in collection.items():
            result[k] = _transform_items(v, transform)
        return result
    else:
        return transform(collection)

import fcntl
import copy
class StackCache(dict):
    cache_file = "/var/tmp/{}.singularity-stack.yml".format(getpass.getuser())

    @classmethod
    def load(cls):
        with open(cls.cache_file, "r") as fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                stacks = yaml.load(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        return stacks

    def __init__(self):
        try:
            self.fd = open(self.cache_file, "r+")
        except FileNotFoundError:
            self.fd = open(self.cache_file, "a+")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            contents = yaml.load(self.fd)
            if contents is None:
                contents = {}
            super(StackCache, self).__init__(contents)
        except:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            raise

    def add(self, name, compose_file=None):
        if compose_file is None:
            self[name]['active'] = True
            return copy.deepcopy(self[name])
        with open(compose_file, 'rb') as f:
            config = _transform_items(yaml.load(f), expandvars)
        version = float(config.get('version', 0))
        if version < 3:
            raise ValueError("Unsupported docker-compose version '{}'".format(version))
        self[name] = copy.deepcopy(config)
        self[name]['active'] = True
        return config

    def remove(self, name):
        self[name]['active'] = False

    def __del__(self):
        try:
            self.fd.truncate(0)
            if len(self) > 0:
                self.fd.seek(0, 0)
                yaml.dump(dict(self.items()), self.fd)
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()

def list_stacks(args):
    """List active stacks"""
    stacks = StackCache.load()
    template = '{:30s} {:30s} {:7s} {:8s}'
    print(template.format('Stack', 'Services', 'Replicas', 'Instance'))
    print(template.format('='*30, '='*30, '='*7, '='*8))
    for name, config in stacks.items():
        if not config.get('active', False):
            continue
        for i, service in enumerate(config['services'].keys()):
            reps = int(config['services'][service].get('deploy', {}).get('replicas', 1))
            label = _instance_name(name, service, 0)
            print(template.format(name if i==0 else '', service, str(reps) if reps > 1 else '', label))
        print(template.format('-'*30, '-'*30, '-'*7, '-'*8))

def _sub_replica(obj, replica):
    pattern = "{{.Task.Slot}}"
    if isinstance(obj, str) and pattern in obj:
        return obj.replace(pattern, str(replica+1))
    else:
        return obj

def _stop(app, name, config):
    try:
        with open(_pid_file(app,name), 'r') as pidfile:
            pid = int(pidfile.read())
            log.debug('Killing process {}'.format(pid))
            try:
                os.kill(pid, signal.SIGINT)
            except:
                pass
            while True:
                try:
                    os.kill(pid, 0)
                    time.sleep(1)
                except Exception as e:
                    break
    except FileNotFoundError:
        pass
    service = config['services'][name]
    replicas = int(service.get('deploy', {}).get('replicas', 1))
    for replica in range(replicas):
        instance = _instance_name(app, name, replica)
        if _instance_running(instance):
            subprocess.check_call(['singularity', 'instance.stop'] + [instance], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            log.info("Stopped instance {} ({}.{}.{})".format(instance, app, name, replica))

def _start_service(app, name, config):
    _stop(app, name, config)
    log.debug('Starting service {}.{}'.format(app, name))
    service = config['services'][name]
    replicas = int(service.get('deploy', {}).get('replicas', 1))
    configs = []
    for replica in range(replicas):
        instance = _instance_name(app, name, replica)
        myconfig = copy.deepcopy(config)
        myconfig['services'][name] = _transform_items(config['services'][name], lambda x: _sub_replica(x, replica))
        configs.append(myconfig)
        image_spec = singularity_command_line(name, myconfig)
        if len(os.path.basename(image_spec[-1]))+len(instance) > 32:
            raise ValueError("image file ({}) and instance name ({}) can have at most 32 characters combined. (singularity 2.4 bug)".format(os.path.basename(image_spec[-1]), instance))
        subprocess.check_call(['singularity', 'instance.start'] + image_spec + [instance])
        log.debug("Started instance {} ({}.{}.{})".format(instance, app, name, replica))
    _start_replica_set(app, name, configs)

def update(args):
    """[Re]start a single service"""
    stacks = StackCache()
    config = stacks[args.name]
    service = config['services'][args.service]
    if args.volume:
        volumes = {v['target']: v for v in map(_parse_volume, service['volumes'])}
        for v in map(_parse_volume, args.volume):
            volumes[v['target']] = v
        service['volumes'] = list(volumes.values())
    del stacks
    _start_service(args.name, args.service, config)

def stop(args):
    """Stop a single service"""
    stacks = StackCache()
    config = stacks[args.name]
    del stacks
    _stop(args.name, args.service, config)

def load(args):
    """load and cache a stack"""
    stacks = StackCache()
    config = stacks.add(args.name, args.compose_file)
    del stacks

def deploy(args):
    """[Re]start all services in the stack"""
    stacks = StackCache()
    config = stacks.add(args.name, args.compose_file)
    del stacks
    app = args.name
    try:
        for name in start_order(config['services']):
            _start_service(app, name, config)
    except Exception as e:
        print('caught {}, shutting down'.format(e)) 
        for name in reversed(list(start_order(config['services']))):
            _stop(app, name, config)
        raise e

def rm(args):
    """Stop all services in the stack"""
    stacks = StackCache()
    app = args.name
    config = stacks[app]
    for name in reversed(list(start_order(config['services']))):
        _stop(app, name, config)
    stacks.remove(app)

def logs(args):
    """View service logs"""
    config = StackCache.load()[args.name]
    if not args.stderr or args.stdout:
        args.stderr = True
        args.stdout = True

    if args.service is None:
        names = sorted(config['services'].keys())
        raise ValueError("I need a single service at the moment")
    else:
        names = [args.service]
    args.exclude = [re.compile(p) for p in args.exclude]

    template = "{{:23s}} {{:{}s}}:{{}} {{:7s}} \u23b8 ".format(max(map(len, names)))

    path = _log_prefix(args.name, args.service)+".json"
    f = open(path, "rb")
    size = os.stat(path).st_size
    if args.follow and args.since is None:
        # start 1 kB from the end of the file
        offset = -min((1024, size))
        f.seek(offset, 2)
        # consume the partial line
        f.readline()
    i = 0
    while True:
       line = f.readline()
       if len(line) == 0:
           if not args.follow:
               break
           else:
               try:
                   time.sleep(0.5)
               except KeyboardInterrupt:
                   break
               continue
       i += 1
       try:
           payload = json.loads(line.decode('utf-8'))
       except json.decoder.JSONDecodeError:
           continue
       if args.since is not None and payload['timestamp'] < args.since:
           continue
       if (not args.stderr and payload['source'] == 'stderr') or (not args.stdout and payload['source'] == 'stdout'):
           continue
       if (args.replica is not None and payload.get('replica', None) != args.replica):
           continue
       if any(p.match(payload['msg']) for p in args.exclude):
           continue
       ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(payload['timestamp'])) + "{:.3f}".format(payload['timestamp'] % 1)[1:]
       sys.stdout.write(template.format(ts, args.service, payload.get('replica', 0), payload['source']))
       print(payload['msg'])

def _ppid(pid):
    # lifted from psutil
    with open('/proc/{}/stat'.format(pid), 'rb') as f:
        data = f.read()
    rpar = data.rfind(b')')
    dset = data[rpar + 2:].split()
    return int(dset[1])

def _executable(pid):
    with open('/proc/{}/cmdline'.format(pid), 'rb') as f:
        data = f.read()
    fields = data.split(b'\0')
    if b'/' in fields[0]:
        return fields[0].split(b'/')[-1]
    else:
        return fields[0]

def _ppid_map():
    # lifted from psutil
    ppids = {}
    for pid in os.listdir('/proc/'):
        if pid.isdigit():
            ppids[int(pid)] = _ppid(int(pid))
    return ppids

def _service_pids(root_pid):
    """Gather PIDs that are descendants of root_pid, but whose direct parent is action-suid. These are processes started with `singularity run`."""
    def _gather_leaves(root_pid, parent_map, child_map):
        if _executable(parent_map[root_pid]) == b'action-suid':
            return [root_pid]
        elif root_pid in child_map:
            return sum([_gather_leaves(c, parent_map, child_map) for c in child_map[root_pid]], [])
        else:
            return []
    parent_map = _ppid_map()
    child_map = {}
    for c, p in parent_map.items():
        if not p in child_map:
            child_map[p] = []
        child_map[p].append(c)
    return _gather_leaves(root_pid, parent_map, child_map)

def _get_environment(app, name):
    """Retrieve the environment variables for a running service"""
    # get PID of service daemon
    with open(_pid_file(app, name), 'rb') as f:
        pid = int(f.read())
    # get PID of a replica process
    children = _service_pids(pid)
    with open('/proc/{}/environ'.format(children[0]), 'rb') as f:
        data = f.read()
    env = dict(os.environ)
    for field in data.split(b'\0'):
        i = field.find(b'=')
        k, v = field[:i], field[i+1:]
        env[b'SINGULARITYENV_'+k] = v
    return env

def exec(args):
    """
    Execute a program in the container of the given service, with its environment variables
    """
    stacks = StackCache()
    config = stacks[args.name]
    del stacks
    instance = _instance_name(args.name, args.service, 0)
    os.execvpe('singularity',
        ['singularity', 'exec', '--cleanenv', 'instance://{}'.format(instance)] +  args.cmd,
        _get_environment(args.name, args.service))

def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(prog='singularity-stack', description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version='singularity-stack {}'.format(__version__))
    parser.add_argument('--debug', action='store_true', default=False)

    subparsers = parser.add_subparsers(help='command help')

    def add_command(f, name=None, needs_name=True):
        if name is None:
            name = f.__name__
        p = subparsers.add_parser(name, help=f.__doc__)
        p.set_defaults(func=f)
        if needs_name:
            p.add_argument('name', help="name of stack (argument passed to `singularity-stack deploy`")
        return p

    add_command(list_stacks, 'list', False)
    p = add_command(deploy)
    p.add_argument('-c', '--compose-file', type=str, default=None)

    p = add_command(load)
    p.add_argument('-c', '--compose-file', type=str, default=None)

    p = add_command(update)
    p.add_argument('service')
    p.add_argument('--volume', default=[], action='append', help='Add or replace a volume mount')

    p = add_command(stop)
    p.add_argument('service')

    p = add_command(exec)
    p.add_argument('service')
    p.add_argument('cmd', nargs='+')

    p = add_command(rm)

    p = add_command(logs)
    def parse_timedelta(value):
        now = datetime.datetime.now()
        if value.endswith('s'):
            return datetime.timedelta(seconds=float(value[:-1]))
        elif value.endswith('m'):
            return datetime.timedelta(seconds=60*float(value[:-1]))
        elif value.endswith('h'):
            return datetime.timedelta(seconds=3600*float(value[:-1]))
        elif value.endswith('d'):
            return datetime.timedelta(days=float(value[:-1]))
        else:
            raise ValueError('Unrecognized time interval "{}"'.format(value))
    def get_timestamp(value):
        try:
            return (datetime.datetime.now() - parse_timedelta(value)).timestamp()
        except ValueError:
            return dateutil.parser.parse(value).timestamp()
    p.add_argument('service', default=None, nargs='?', type=str, help='Dump logs for this service only')
    p.add_argument('replica', default=None, nargs='?', type=int, help='Dump logs for this replica only')
    p.add_argument('-f', '--follow', default=False, action="store_true", help='Follow log output')
    p.add_argument('--since', default=None, type=get_timestamp, help='Show logs since relative time (e.g. 1d, 1.2h, 5m, 30s)')
    p.add_argument('--stdout', default=False, action="store_true", help='Dump only stdout')
    p.add_argument('--stderr', default=False, action="store_true", help='Dump only stderr')
    p.add_argument('-x', '--exclude', default=[], action="append", help='Exclude lines matching this pattern')

    subvolume = subparsers.add_parser('volume', help=_init_volume.__doc__).add_subparsers()
    p = subvolume.add_parser('init', description='initialize a singularity-stack volume by copying a path from the image to the host filesystem.')
    p.set_defaults(func=init_volume_cmd)
    p.add_argument('image', help='singularity image')
    p.add_argument('source', help='source path in image')
    p.add_argument('dest', help='destination path in host filesystem')
    p.add_argument('-f', '--force', default=False, action='store_true', help='overwrite destination path if it exists')

    opts = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)s (%(process)d) %(funcName)s() %(levelname)s %(message)s',
                        level='DEBUG' if opts.debug else 'INFO')
    opts.func(opts)

if __name__ == "__main__":
    main()

