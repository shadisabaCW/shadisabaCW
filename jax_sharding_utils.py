import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding

def setup_distributed_mesh(devices):
    mesh = Mesh(devices, ('data', 'model'))
    sharding = NamedSharding(mesh, P('data', 'model'))
    return sharding
