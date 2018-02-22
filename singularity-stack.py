#!/usr/bin/env python3

"""
Deploy a Docker application with Singularity, in a manner similar to `docker-stack`
"""

import yaml
import sys, os
import re
import time
import subprocess
import getpass

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
        basename, _ = name.split(':')
    
    cwd = os.environ.get('SINGULARITY_CACHEDIR', os.getcwd())
    image_path = os.path.join(cwd, basename+'.simg')
    if not os.path.isfile(image_path):
        subprocess.check_call(['singularity', 'pull', 'docker://{}'.format(name)], cwd=cwd)
    return image_path

def singularity_command_line(name, service):
    """
    Start a Singularity instance from the given Docker service definition
    """
    # TODO:
    # tmpfs
    # entrypoint
    # ports
    # expose
    if not 'image' in service:
        raise ValueError("service '{}' is missing an 'image' key".format(name))
    cmd = ['--contain']
    for volume in service.get('volumes', []):
        cmd += ['-B', volume]
    
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

def _run(app, name, service):
    """
    Fork a daemon process that starts the service process in a Singularity
    instance, redirects its stdout/stderr to files, and restarts it on failure
    according to the given `restart_policy`
    """
    if 'restart' in service:
        raise ValueError("restart is not supported. use deploy instead")
    instance = _instance_name(app, name)
    # NB: we assume that the default entrypoint simply calls exec on arguments
    # it does not recognize
    cmd = ['singularity', 'run', 'instance://'+instance]
    if 'command' in service:
        assert isinstance(service['command'], list)
        cmd += service['command']
    
    env = service.get('environment', {})
    
    restart_policy = service.get('deploy', {}).get('restart_policy', {})
    if not restart_policy.get('condition', False) in {'on-failure', False}:
        raise ValueError("unsupported restart condition '{}'".format(restart_policy['condition']))
    
    max_attempts = 0
    if restart_policy.get('condition','no') == 'on-failure':
        max_attempts = restart_policy.get('max_attempts', sys.maxsize)
    delay = _parse_duration(restart_policy.get('delay', ''))
    
    stderr = open(_log_prefix(app, name)+".stderr", "wb")
    stdout = open(_log_prefix(app, name)+".stdout", "wb")
    
    log = lambda s: stderr.write((s+'\n').encode('utf-8'))
    
    if os.fork() == 0:
        ret = 0
        for _ in range(-1, max_attempts):
            if not _instance_running(instance):
                log('{} instance is gone!'.format(instance))
                sys.exit(1)
            ret = subprocess.call(cmd, env=env, stdout=stdout, stderr=stderr)
            if ret == 0:
                break
            else:
                log('sleeping {} before restart')
                time.sleep(delay)
        if ret != 0:
            log('failed permanently\n')
        else:
            log('exited cleanly\n')
        stderr.close()
        stdout.close()
        sys.exit(ret)

def up(app, name, service):
    instance = _instance_name(app, name)
    if _instance_running(instance):
        subprocess.check_call(['singularity', 'instance.stop'] + [instance])
    image_spec = singularity_command_line(name, service)
    # raise ValueError(image_spec)
    subprocess.check_call(['singularity', 'instance.start'] + image_spec + [instance])
    _run(app, name, service)

def down(app, name, service):
    instance = _instance_name(app, name)
    subprocess.check_call(['singularity', 'instance.stop', instance])

if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('command', choices=['deploy', 'rm'])
    parser.add_argument('name')
    parser.add_argument('-c', '--compose-file', type=str, default='docker-compose.yml')
    opts = parser.parse_args()
    
    with open(opts.compose_file, 'rb') as f:
        config = yaml.load(f)
    version = config.get('version', None)
    if version != '3':
        raise ValueError("Unsupported docker-compose version '{}'".format(version))
    
    services = config.get('services', dict())
    
    if opts.command == 'deploy':
        for key in start_order(services):
            up(opts.name, key, services[key])
    elif opts.command == 'rm':
        for key in reversed(list(start_order(services))):
            down(opts.name, key, services[key])
    

