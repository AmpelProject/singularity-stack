
from singularity_stack import ConstraintEvaluator
import socket
import pytest

def test_eval():
    ce = ConstraintEvaluator()
    assert ce('foo-1') == 'foo-1'
    assert ce('node.hostname == {}'.format(socket.gethostname().split('.')[0]))
    assert not ce('node.hostname == {}-notaname'.format(socket.gethostname().split('.')[0]))
    
    with pytest.raises(ValueError):
        ce('node.hostname > 1')
    with pytest.raises(ValueError):
        ce('node.label.foo == blah')
