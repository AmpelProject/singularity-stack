#!/usr/bin/env python3

"""
Deploy a Docker application with Singularity, in a manner similar to `docker-stack`
"""

import yaml
import daemon
import sys, os, stat, shutil
import warnings
import re
import time
import subprocess
import getpass
import tempfile
import asyncio

__version__ = "0.2"

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
        raise ValueError("Secrets file '{}' should be readable only by its owner")
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
    for volume in service.get('volumes', []):
        if isinstance(volume, str):
            if ':' in volume:
                source, dest = volume.split(':')
            else:
                source, dest = volume, volume
            if source.startswith('.') or source.startswith('/'):
                voltype = 'bind'
            else:
                voltype = 'volume'
            volume = dict(type=voltype, source=source, target=dest)
            
        if isinstance(volume, dict):
            if volume['type'] == 'volume':
                warnings.warn("Volume '{}' will be treated as a bind mount".format(volume['source']))
                if not os.path.isdir(volume['source']):
                    raise ValueError("Volume source path '{}' does not exist".format(volume['source']))
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

def _instance_name(*args):
    return '.'.join(args)

def _log_prefix(*args):
    return "/var/tmp/{}.singularity.{}".format(getpass.getuser(), _instance_name(*args))

def _instance_running(name):
    # FIXME: make this check for an exact match
    ret = subprocess.call(['singularity', 'instance.list', name],
        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return ret == 0

def _run(app, name, config):
    """
    Fork a daemon process that starts the service process in a Singularity
    instance, redirects its stdout/stderr to files, and restarts it on failure
    according to the given `restart_policy`
    """
    service = config['services'][name]
    if 'restart' in service:
        raise ValueError("restart is not supported. use deploy instead")
    instance = _instance_name(app, name)
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
    if not restart_policy.get('condition', False) in {'on-failure', False}:
        raise ValueError("unsupported restart condition '{}'".format(restart_policy['condition']))
    
    max_attempts = 0
    if restart_policy.get('condition','no') == 'on-failure':
        max_attempts = restart_policy.get('max_attempts', sys.maxsize)
    delay = _parse_duration(restart_policy.get('delay', ''))
    
    if os.fork() != 0:
        return
    
    stderr = open(_log_prefix(app, name)+".stderr", "wb")
    stdout = open(_log_prefix(app, name)+".stdout", "wb")
    
    with daemon.DaemonContext(detach_process=True, stderr=stderr, stdout=stdout):
        ret = 0
        for _ in range(-1, max_attempts):
            if not _instance_running(instance):
                print('{} instance is gone!'.format(instance))
                sys.exit(1)
            ret = subprocess.call(cmd, env=env, stdout=stdout, stderr=stderr)
            if ret == 0:
                break
            else:
                print('sleeping {} before restart'.format(delay))
                time.sleep(delay)
        if ret != 0:
            print('failed permanently\n')
        else:
            print('exited cleanly\n')
        for temp in service.get('_tempfiles', []):
            if hasattr(temp, 'name'):
                temp = getattr(temp, 'name')
            print('removing {}'.format(temp))
            if os.path.isdir(temp):
                shutil.rmtree(temp)
            else:
                os.unlink(temp)
                
        sys.exit(ret)

def _load_services(args):
    with open(args.compose_file, 'rb') as f:
        config = yaml.load(f)
    version = float(config.get('version', 0))
    if version < 3:
        raise ValueError("Unsupported docker-compose version '{}'".format(version))
    return config

def deploy(args):
    config = _load_services(args)
    app = args.name
    try:
        for name in start_order(config['services']):
            service = config['services'][name]
            instance = _instance_name(app, name)
            if _instance_running(instance):
                subprocess.check_call(['singularity', 'instance.stop'] + [instance])
            image_spec = singularity_command_line(name, config)
            if len(os.path.basename(image_spec[-1]))+len(instance) > 32:
                raise ValueError("image file ({}) and instance name ({}) can have at most 32 characters combined. (singularity 2.4 bug)".format(os.path.basename(image_spec[-1]), instance))
            subprocess.check_call(['singularity', 'instance.start'] + image_spec + [instance])
            _run(app, name, config)
    except Exception as e:
        print('caught {}, shutting down'.format(e)) 
        for name in reversed(list(start_order(config['services']))):
            instance = _instance_name(app, name)
            if _instance_running(instance):
                subprocess.check_call(['singularity', 'instance.stop'] + [instance])
        raise e

def rm(args):
    config = _load_services(args)
    app = args.name
    for name in reversed(list(start_order(config['services']))):
        instance = _instance_name(app, name)
        if _instance_running(instance):
            subprocess.check_call(['singularity', 'instance.stop'] + [instance])

def logs(args):
    async def readline(f):
        while True:
            line = f.readline()
            if line:
                return line
            await asyncio.sleep(0.01)
    
    async def _tail(app, name, suffix, linetag=''):
        path = "{}.{}".format(_log_prefix(app, name),suffix)
        offset = -min((1024, os.stat(path).st_size))
        with open(path, "rb") as f:
            # start 1 kB from the end of the file
            f.seek(offset,2)
            # consume the partial line
            f.readline()
    
            while True:
                line = await readline(f)
                sys.stdout.write(linetag)
                sys.stdout.write(line.decode())
    
    loop = asyncio.get_event_loop()
    
    services = _load_services(args)
    if not args.stderr or args.stdout:
        args.stderr = True
        args.stdout = True
    
    if args.service is None:
        names = sorted(services.keys())
    else:
        names = [args.service]
    
    template = "({{:{}s}} {{:6s}}) ".format(max(map(len, names)))
    
    for name in names:
        if args.stderr:
            asyncio.ensure_future(_tail(args.name, name, "stderr", template.format(name,"stderr")))
        if args.stdout:
            asyncio.ensure_future(_tail(args.name, name, "stdout", template.format(name,"stdout")))
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        return
    finally:
        for task in asyncio.Task.all_tasks():
            task.cancel()
        loop.run_until_complete(asyncio.gather(*asyncio.Task.all_tasks(), return_exceptions=True))
        loop.close()

def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(prog='singularity-stack', description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version='singularity-stack {}'.format(__version__))
    parser.add_argument('-c', '--compose-file', type=str, default='docker-compose.yml')
    
    subparsers = parser.add_subparsers(help='command help')
    
    subparsers.add_parser('deploy') \
        .set_defaults(func=deploy)
    subparsers.add_parser('rm') \
        .set_defaults(func=rm)
    
    parser_logs = subparsers.add_parser('logs')
    parser_logs.set_defaults(func=logs)
    parser_logs.add_argument('-s', '--service', default=None, type=str, help='Dump logs for this service only')
    parser_logs.add_argument('--stdout', default=False, action="store_true", help='Dump only stdout')
    parser_logs.add_argument('--stderr', default=False, action="store_true", help='Dump only stderr')

    parser.add_argument('name')

    opts = parser.parse_args()
    opts.func(opts)

if __name__ == "__main__":
    main()
