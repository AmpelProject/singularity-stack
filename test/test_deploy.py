
import pytest
import asyncio

from singularity_stack import ReplicaSetController, ReplicaRunner

@pytest.fixture
def ohai_config():
    return {
      'version': '3',
      'services': {
        'ohai': {
            'image': 'alpine:3.6',
            'command': ["echo", "ohai"]
        }
      }
    }

@pytest.fixture
def sleeper_config():
    return {
      'version': '3',
      'services': {
        'ohai': {
            'image': 'alpine:3.6',
            'command': ["sleep", "3600"]
        }
      }
    }

def test_start_instance(ohai_config):
    instance, config = ReplicaSetController._start_instance('test', 'ohai', 0, ohai_config)
    assert ReplicaSetController._instance_running(instance)
    assert ReplicaSetController._stop_instance(instance)
    assert not ReplicaSetController._instance_running(instance)

def test_run_service(ohai_config):
    class NullHandler:
        def __init__(self):
            self.records = []
        def format(self, record):
            return record
        def emit(self, record):
            self.records.append(record)
    handler = NullHandler()

    instance, config = ReplicaSetController._start_instance('test', 'ohai', 0, ohai_config)
    r = ReplicaRunner('test', 'ohai', 0, ohai_config, handler)
    ret = asyncio.get_event_loop().run_until_complete(r.wait())

    stdout = [d for d in handler.records if d['source'] == 'stdout']
    stderr = [d for d in handler.records if d['source'] == 'stderr']
    nanny = [d for d in handler.records if d['source'] == 'nanny']

    assert nanny
    assert not stderr

    assert len(stdout) == 1
    assert stdout[0]['msg'] == 'ohai'

    assert ReplicaSetController._stop_instance(instance)



def test_version():
    from singularity_stack import singularity_version, Version
    assert Version(2,5,0) < singularity_version() < Version(4,0,0)

    
    