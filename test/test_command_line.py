
import pytest

import tempfile
import os
import copy
import shutil
import subprocess
from singularity_stack import singularity_command_line, singularity_version

@pytest.fixture
def config():
    config = {
      'version': '3',
      'services': {
        'foo': {
            'image': 'alpine:3.6'
        }
      }
    }
    yield config
    for temp in config['services']['foo'].get('_tempfiles', []):
        if hasattr(temp, 'name'):
            temp = getattr(temp, 'name')
        if os.path.isdir(temp):
            shutil.rmtree(temp)
        else:
            os.unlink(temp)

def test_missing_image(config):
    del config['services']['foo']['image']
    with pytest.raises(ValueError):
        singularity_command_line('foo', config)

def test_mount_volume(config):
    with tempfile.TemporaryDirectory() as tmp:
        config['services']['foo']['volumes'] = [
            {
                'type': 'volume',
                'source': tmp,
                'target': '/media'
            }
        ]
        assert not os.path.isdir(os.path.join(tmp, 'cdrom'))
        singularity_command_line('foo', config)
        assert os.path.isdir(os.path.join(tmp, 'cdrom'))

def test_mount_tmpfs(config):
    config['services']['foo']['volumes'] = [
        {
            'type': 'tmpfs',
            'target': '/media'
        }
    ]
    assert '_tempfiles' not in config['services']['foo']
    cmd = singularity_command_line('foo', config)
    assert '_tempfiles' in config['services']['foo']
    print(cmd)
    subprocess.check_call(['singularity', 'exec'] + cmd + ['touch', '/media/foo'])
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(['singularity', 'exec'] + cmd + ['touch', '/foo'])
    subprocess.check_call(['singularity', 'exec'] + cmd + ['cat', '/media/foo'])

def test_mount_bind(config):
    with tempfile.TemporaryDirectory() as tmp:
        config['services']['foo']['volumes'] = [
            {
                'type': 'bind',
                'source': tmp,
                'target': '/media'
            }
        ]
        cmd = singularity_command_line('foo', config)
        subprocess.check_call(['singularity', 'exec'] + cmd + ['touch', '/media/foo'])
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['singularity', 'exec'] + cmd + ['touch', '/foo'])
        assert 'foo' in os.listdir(tmp)

def test_mount_unsupported(config):
    config['services']['foo']['volumes'] = [{'type': None}]
    with pytest.raises(ValueError):
        singularity_command_line('foo', config)

def test_bind_secrets(config):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b'blerh')
        tmp.flush()
        tmp.seek(0)
        assert tmp.read() == b'blerh'
        config['services']['foo']['secrets'] = ['seekrit']
        config['secrets'] = {
            'seekrit': {
                'file': tmp.name
            }
        }
        cmd = singularity_command_line('foo', config)
        assert subprocess.check_output(['singularity', 'exec'] + cmd + ['cat', '/run/secrets/seekrit']) == b'blerh'

def test_extra_hosts(config):
    config['services']['foo']['extra_hosts'] = ['flerp:1.2.3.4']
    cmd = singularity_command_line('foo', config)
    assert '_tempfiles' in config['services']['foo']
    record = subprocess.check_output(['singularity', 'exec'] + cmd + ['nslookup', 'flerp']).decode().strip().split('\n')[-1]
    assert record == 'Address 1: 1.2.3.4 flerp'

def test_parse_volume():
    from singularity_stack import _parse_volume
    
    assert _parse_volume('/foo:/bar') == {'type': 'bind', 'source': '/foo', 'target': '/bar'}
    assert _parse_volume('foo:/bar') == {'type': 'volume', 'source': 'foo', 'target': '/bar'}
    passthrough = {'type': 'volume', 'source': 'foo', 'target': '/bar'}
    assert _parse_volume(passthrough) == passthrough
    with pytest.raises(TypeError):
        _parse_volume(['thing'])

@pytest.fixture
def restore_env():
    prev = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(**prev)

def test_image_path(tmpdir_factory, restore_env):
    from singularity_stack import singularity_image
    
    tmp = tmpdir_factory.mktemp("singularity-cachedir")
    os.environ['SINGULARITY_CACHEDIR'] = str(tmp)
    image_path = tmp/"alpine-3.6.simg"
    with open(image_path, 'w'):
        pass
    assert singularity_image('alpine:3.6') == image_path
