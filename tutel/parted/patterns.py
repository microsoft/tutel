# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .solver import register_primitive

def is_replicas(dim):
  return dim == -1

def is_partition(dim):
  return dim >= 0

@register_primitive("BAR")
def primitive_pass_through(sess, node, output_dim, group_size, rank):
  if not is_replicas(output_dim) and not is_partition(output_dim):
    return
  source_dims, num_partitions = node.parser.emit_dims_by_id(output_dim)

  if is_replicas(output_dim) and num_partitions == 0:
    yield (0, source_dims, {})
    return

  connectors = dict([(inp, sess.backend.link('$', -1, None, is_param=(node.inputs[inp].op_type == "param"))) for inp in source_dims if is_replicas(source_dims[inp])])
  yield (0, source_dims, connectors)

@register_primitive("FAR")
def primitive_fwd_allreduce_sum(sess, node, output_dim, group_size, rank):
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
    connectors = dict([(inp, sess.backend.link('$', -1, None, is_param=(node.inputs[inp].op_type == "param"))) for inp in source_dims if is_replicas(source_dims[inp])])
    connectors[''] = sess.backend.link('$', None, -1)
    yield (i, source_dims, connectors)

@register_primitive("RS")
def primitive_fwd_reduce_scatter_sum(sess, node, output_dim, group_size, rank):
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
    connectors = dict([(inp, sess.backend.link('$', -1, None, is_param=(node.inputs[inp].op_type == "param"))) for inp in source_dims if is_replicas(source_dims[inp])])
    connectors[''] = sess.backend.link('$', None, output_dim)
    yield (i, source_dims, connectors)

@register_primitive("SPLIT")
def primitive_fwd_spatial_split(sess, node, output_dim, group_size, rank):
  if not is_partition(output_dim):
    return
  source_dims, num_partitions = node.parser.emit_dims_by_id(-1)
  assert num_partitions == 0, "It is unexpected that certain input is parted."
  connectors = dict([('', sess.backend.link('$', -1, output_dim))])
  yield (0, source_dims, connectors)

@register_primitive("AG")
def primitive_fwd_all_gather(sess, node, output_dim, group_size, rank):
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
    connectors = dict([(inp, sess.backend.link('$', -1, None, is_param=(node.inputs[inp].op_type == "param"))) for inp in source_dims if is_replicas(source_dims[inp])])
    connectors[''] = sess.backend.link('$', rank, -1)
    yield (i, source_dims, connectors)

@register_primitive("A2A")
def primitive_alltoall(sess, node, output_dim, group_size, rank):
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
      connectors = dict([(inp, sess.backend.link('$', -1, None, is_param=(node.inputs[inp].op_type == "param"))) for inp in source_dims if is_replicas(source_dims[inp])])
      connectors[''] = sess.backend.link('$', i, output_dim)
      yield (i, source_dims, connectors)
    except NotImplementedError:
      continue

@register_primitive("ZERO")
def primitive_zero(sess, node, output_dim, group_size, rank):
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
      has_params, connectors[inp] = True, sess.backend.link('$', -2, -1, output_shape=node.inputs[inp].shape)
    else:
      connectors[inp] = sess.backend.link('$', -1, None, is_param=(node.inputs[inp].op_type == "param"))
  if not has_params:
    return
  yield (0, source_dims, connectors)
