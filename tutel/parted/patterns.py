# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .solver import register_primitive

def is_replicas(dim):
  return dim == -1

def is_partition(dim):
  return dim >= 0

@register_primitive("BAR")
def primitive_pass_through(node, output_dim, group_size, rank):
  if not is_replicas(output_dim) and not is_partition(output_dim):
    return
  source_dims, num_partitions = node.parser.emit_dims_by_id(output_dim)

  if is_replicas(output_dim) and num_partitions == 0:
    yield (0, source_dims, {})
    return

  connectors = dict([(inp, f'warp_bwd_allreduce($, {node.inputs[inp].op_type == "param"})') for inp in source_dims if is_replicas(source_dims[inp])])
  yield (0, source_dims, connectors)

@register_primitive("FAR")
def primitive_fwd_allreduce_sum(node, output_dim, group_size, rank):
  if not is_replicas(output_dim):
    return
  if node.parser.reduce_type != '+':
    return

  for i, ax in enumerate(node.parser.get_reduce_axes()):
    if rank is not None and i != rank:
      continue
    try:
      source_dims, num_partitions = node.parser.emit_dims_by_name(ax)
    except NotImplementedError:
      continue
    assert num_partitions > 0, "It is unexpected that no certain input is parted."
    connectors = dict([(inp, f'warp_bwd_allreduce($, {node.inputs[inp].op_type == "param"})') for inp in source_dims if is_replicas(source_dims[inp])])
    connectors[''] = f'C.PrimFwdAllreduce.apply(default_group, $)'
    yield (i, source_dims, connectors)

@register_primitive("RS")
def primitive_fwd_reduce_scatter_sum(node, output_dim, group_size, rank):
  if not is_partition(output_dim):
    return
  if node.parser.reduce_type != '+':
    return

  for i, ax in enumerate(node.parser.get_reduce_axes()):
    if rank is not None and i != rank:
      continue
    try:
      source_dims, num_partitions = node.parser.emit_dims_by_name(ax)
    except NotImplementedError:
      continue
    assert num_partitions > 0, "It is unexpected that no certain input is parted."
    connectors = dict([(inp, f'warp_bwd_allreduce($, {node.inputs[inp].op_type == "param"})') for inp in source_dims if is_replicas(source_dims[inp])])
    connectors[''] = f'C.PrimReducescatter.transform(default_group, $, {output_dim})'
    yield (i, source_dims, connectors)

@register_primitive("SPLIT")
def primitive_fwd_spatial_split(node, output_dim, group_size, rank):
  if not is_partition(output_dim):
    return
  source_dims, num_partitions = node.parser.emit_dims_by_id(-1)
  assert num_partitions == 0, "It is unexpected that certain input is parted."
  connectors = dict([('', f'C.PrimSpatialSplit.transform(default_group, $, {output_dim})')])
  yield (0, source_dims, connectors)

@register_primitive("AG")
def primitive_fwd_all_gather(node, output_dim, group_size, rank):
  if not is_replicas(output_dim):
    return
  for i in range(len(node.shape)):
    if rank is not None and i != rank:
      continue
    try:
      if node.shape[i] % group_size != 0:
        continue
      source_dims, num_partitions = node.parser.emit_dims_by_id(i)
    except NotImplementedError:
      continue
    if num_partitions == 0: # Handled by fwd_pass_through as well
      continue
    connectors = dict([(inp, f'warp_bwd_allreduce($, {node.inputs[inp].op_type == "param"})') for inp in source_dims if is_replicas(source_dims[inp])])
    connectors[''] = f'C.PrimAllgather.transform(default_group, $, {rank})'
    yield (i, source_dims, connectors)

@register_primitive("A2A")
def primitive_alltoall(node, output_dim, group_size, rank):
  if not is_partition(output_dim):
    return
  shape = node.shape
  if len(shape) < 2 or shape[output_dim] % group_size != 0:
    return
  for i in range(len(node.shape)):
    if rank is not None and i != rank:
      continue
    if shape[i] % group_size != 0 or output_dim == i:
      continue
    try:
      source_dims, num_partitions = node.parser.emit_dims_by_id(i)
      connectors = dict([(inp, f'warp_bwd_allreduce($, {node.inputs[inp].op_type == "param"})') for inp in source_dims if is_replicas(source_dims[inp])])
      connectors[''] = f'C.PrimAllToAll.transform(default_group, $, input_dim={i}, output_dim={output_dim})'
      yield (i, source_dims, connectors)
    except NotImplementedError:
      continue

@register_primitive("ZERO")
def primitive_zero(node, output_dim, group_size, rank):
  if not is_partition(output_dim):
    return
  source_dims, num_partitions = node.parser.emit_dims_by_id(output_dim)
  if num_partitions == 0:
    return
  has_params, connectors = False, {}
  for inp in source_dims:
    if not is_replicas(source_dims[inp]):
      continue
    if node.inputs[inp].op_type == 'param':
      source_dims[inp] = -2
      has_params, connectors[inp] = True, f'C.PrimAllgather.zero_param(default_group, $, {node.inputs[inp].shape})'
    else:
      connectors[inp] = f'warp_bwd_allreduce($, {node.inputs[inp].op_type == "param"})'
  if not has_params:
    return
  yield (0, source_dims, connectors)
