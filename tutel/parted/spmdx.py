# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys, json, re
import tempfile
import copy
import inspect
import logging
import importlib

from . import solver
from . import patterns

logging.basicConfig(level=logging.INFO)
session = None

def init(backend_name):
  global session
  if session is not None:
    raise Exception('Function `init()` can be only invoked once.')
  if not re.match('^[a-zA-Z0-9]+$', backend_name):
    raise Exception('Only letters and digits are allowed for backend_name, get: %s' % backend_name)
  session = init
  session.backend = importlib.import_module('..backend.%s.config' % backend_name, __name__)
  session.is_strict_fmt = int(os.environ.get('STRICT_FMT', 0)) > 0
  session.ptype = os.environ.get('PTYPE', '')
  session.custom_dict = dict()

  manual_config = os.environ.get('CONFIG', '')
  manual_config = json.loads(manual_config) if manual_config else {}
  manual_config = dict([(x, manual_config[x] if isinstance(manual_config[x], int) else manual_config[x][0]) for x in manual_config])
  session.manual_config = manual_config
  try:
    extra = importlib.import_module('..backend.%s' % backend_name, __name__)
  except:
    extra = None
  return extra

def new_dependency(header_content, depends=[]):
  header_content = header_content.strip() + '\n'
  depends = depends if isinstance(depends, list) else [depends]
  return {"data": header_content, "depends": depends}

def product(arrlist):
  result = 1
  for x in arrlist: result *= int(x)
  return result

class Mapper2D:
  def __init__(self, item):
    def split_dim(item):
      parts = item.replace(')', '(').split('(')
      for i in range(len(parts)):
        if i % 2 == 0:
          for x in parts[i]:
            if x.strip():
              yield x
        else:
          x = [x for x in parts[i] if x.strip()]
          yield x if len(x) > 1 else x[0]

    iter = split_dim(item)
    self.id2ax = [x for x in iter]
    self.ax2id = dict([(x, i) for i, x in enumerate(self.id2ax) if isinstance(x, str) and x != '*'])
    for i, x in enumerate(self.id2ax):
      if not isinstance(x, str):
        for j, ax in enumerate(x):
          self.ax2id[ax] = (i, j)

class Parser:
  def __init__(self, irs):
    left, rights = irs.split('=')
    reduce_type = ''
    if left[-1] in ('+', '<', '>', '[', ']'):
      left, reduce_type = left[:-1], left[-1]

    self.reduce_type = reduce_type
    self.left = Mapper2D(left)
    self.rights = [Mapper2D(x) for x in rights.split(',')]
    self.num_inputs = len(self.rights)

  def get_leading_target(self, target):
    return target if isinstance(target, str) else target[0]

  def get_reduce_axes(self):
    reduce_axes = set()
    for right in self.rights:
      for k in right.ax2id:
        if k not in self.left.ax2id:
          reduce_axes.add(k)
    return reduce_axes

  def emit_dims_by_name(self, ax_name):
    if ax_name == '*':
      raise NotImplementedError()
    target_ax = self.get_leading_target(ax_name)
    source_dims, parted = dict(), 0
    for i, right in enumerate(self.rights):
      if target_ax not in right.ax2id:
        source_dims[i] = -1
        continue
      ids = right.ax2id[target_ax]
      if isinstance(ids, int):
        source_dims[i] = ids
      elif ids[1] == 0:
        source_dims[i] = ids[0]
      else:
        raise NotImplementedError()
      parted += 1
    return source_dims, parted

  def emit_dims_by_id(self, output_dim):
    if output_dim == -1:
      return dict([(i, -1) for i in range(self.num_inputs)]), 0
    if output_dim == -2 or self.left.id2ax[output_dim] == '*':
      raise NotImplementedError()
    if output_dim >= 0:
      return self.emit_dims_by_name(self.left.id2ax[output_dim])
    raise NotImplementedError()


class Program:
  def __init__(self, code, kwargs):
    self.code = code
    self.kwargs = kwargs

  def save(self, path):
    with open(path, 'w') as fp:
      fp.write('# Copyright (c) Microsoft Corporation.\n')
      fp.write('# Licensed under the MIT license.\n\n')
      fp.write(self.code)

  def execute(self, save_file_path=None):
    is_tempfile = save_file_path is None
    if is_tempfile:
      save_file_path = tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir(), suffix='.py').name

    def remove_file(filenames):
      if isinstance(filenames, str):
        filenames = [filenames]
      for filename in filenames:
        try:
          os.unlink(filename)
        except FileNotFoundError:
          pass

    remove_file(save_file_path)

    model_program = self.code
    glob_size = self.kwargs['total_nodes']
    device_type = self.kwargs['device_type']
    group_size = self.kwargs['spmd_nodes']

    with open(save_file_path, 'w') as fp:
      fp.write(model_program)

    log_file = tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir(), suffix='.log').name
    os.environ['CONFIG_STORE_PATH'] = log_file
    remove_file(log_file)
    os_command = session.backend.get_execute_cmd(group_size, glob_size, device_type, save_file_path)

    try:
      result = ''
      logging.info('Executing: %s' % os_command)
      assert 0 == os.system(os_command), f"Failed to execute command: {os_command}"
      with open(log_file, 'r') as fp:
        result = fp.read().strip()
        result = json.loads(result)
    except:
      import traceback
      print(traceback.format_exc())
      print(result)
      result = {}
    if is_tempfile:
      remove_file(save_file_path)
    return result

class Custom:
  __t_builtins__ = dict()
  __t_ids__ = dict()
  __t_ops__ = dict()

  def __init__(self, data, fw_ops=None, inputs=None, op_name=None, shape_fn=None, flops=None, depends=[]):
    self.op_type = op_name or inspect.currentframe().f_back.f_code.co_name
    if not re.match('^[a-zA-Z0-9]+$', self.op_type):
      self.op_type = 'Custom'
    assert self.op_type[0].isupper(), f'The leading charactor of the operator name must be uppercase letter (received: "{self.op_type}").'
    rank_dict = (Custom.__t_ops__ if self.op_type != 'Builtin' else Custom.__t_builtins__) if self.op_type != 'Id' else Custom.__t_ids__

    rank_dict[self] = len(rank_dict)
    self.name = f'{self.op_type[0].lower()}{self.op_type[1:]}{rank_dict[self]}'
    self.depends = depends if isinstance(depends, list) else [depends]

    if fw_ops is not None:
      self.fw_ops = fw_ops.replace('@@', '')

    if inputs is None:
      assert self.fw_ops is not None, 'At least one property in "fw_ops" and inputs should be specified.'
      fw_ops = fw_ops.split('@@')
      input_names = []
      for x in range(1, len(fw_ops), 2):
        if fw_ops[x] not in input_names:
          input_names.append(fw_ops[x])
      self.inputs = [session.custom_dict[x] for x in input_names]
    else:
      self.inputs = inputs

    self.outputs = []
    self.data = data

    if isinstance(data, dict):
      self.op_type = 'data'
      if data["is_param"]:
        self.name += '_'
        self.op_type = 'param'
      self.inputs = []

      self.shape = data["shape"]
      self.dtype = data["dtype"]
      self.flops = flops or 0
    else:
      self.op_type = 'compute'
      self.parser = Parser(data)

      if shape_fn is not None:
        self.shape, self.dtype = shape_fn(self.inputs)
      else:
        try:
          infershape = dict()
          for i, x in enumerate(self.parser.rights):
            for ax in x.ax2id:
              infershape[ax] = self.inputs[i].shape[x.ax2id[ax]]
          self.shape = [infershape[x] if not isinstance(x, list) else product([infershape[y] for y in x]) for x in self.parser.left.id2ax]
          self.dtype = self.inputs[0].dtype
        except:
          raise Exception(f'Cannot auto-infershape for op {self.name} due to unknown dimension size by tensor format: {self.data}')
      # logging.info(f'Shape dict of {self.name} = {self.shape}:{self.dtype}')

      if flops is None:
        self.flops = product(self.shape)
        if self.parser.reduce_type:
          infershape = dict()
          for i, x in enumerate(self.parser.rights):
            for ax in x.ax2id:
              if isinstance(ax, str):
                infershape[ax] = self.inputs[i].shape[x.ax2id[ax]]
          self.flops *= product([infershape[x] for x in self.parser.get_reduce_axes()])
          self.flops <<= 1
      else:
         self.flops = flops

    assert self.name not in session.custom_dict, f"Node with name `{self.name}` has already existed in current session."
    session.custom_dict[self.name] = self

  def __del__(self):
    try:
        session.custom_dict.pop(self.name)
    except:
        pass

  def update_config(self, parent, **kwargs):
    if parent is not None and parent not in self.outputs:
      self.outputs.append(parent)
    node_name = self.name
    if kwargs['spmd_nodes'] == 1:
      self.config = -1
    elif session.ptype == 'dp':
      self.config = -1 if self.op_type == 'param' else 0
    elif session.ptype == 'zero':
      self.config = -2 if self.op_type == 'param' else 0
    elif node_name in session.manual_config:
      self.config = session.manual_config[node_name]
    for input in self.inputs:
      input.update_config(self, **kwargs)

  def __str__(self):
    return f'@@{self.name}@@'

  def numel(self):
    return int(product(self.shape))

  def parse_inputs(self):
    if isinstance(self.data, dict):
      return []
    results, patt = [], self.data
    while True:
      pos = re.search(r'\b[a-z][a-zA-Z0-9_]*\b', patt)
      if not pos:
        break
      results += [patt[pos.start():pos.end()]]
      patt = patt[pos.end() + 1:]
    return results

  def get_leading_dim(self):
    return [i for i, x in enumerate(self.shape) if x > 1][0]

  def get_input_by_name(self, name):
    for inp in self.inputs:
      if inp.name == name:
        return inp
    raise Exception(f'Node input with name `{name}` not found!')

  def autotune(self, config_file=None, **kwargs):
    config = Config.load_from_file(config_file)
    if config:
        return config
    kwargs, results = optimize(self, **kwargs)
    valid_configs = [sol for dim, sol in results if sol is not None]
    if not valid_configs:
      raise Exception('No valid configuration found!')
    best_time, best_config = min(valid_configs)
    config = Config.create(best_config, kwargs, best_time)
    if config_file is not None:
      config.save(config_file)
    return config

  def articulare_analyse(self):
    low, dfn, cut = dict(), dict(), dict()
    pcnt, root, st = [0], self, []

    ##### Mask Articulation Points
    def mask_dfs(u):
      tot = 0
      st.append(u)
      pcnt[0] += 1
      dfn[u] = low[u] = pcnt[0]

      for v in u.inputs + u.outputs:
        # Assume every param tensor is unshared
        if v.op_type == 'param':
          continue
        if v not in dfn:
          tot += 1
          mask_dfs(v)
          low[u] = min(low[u], low[v])
          if ((u == root and tot > 1) or (u != root and low[v] >= dfn[u])):
            cut[u] = cut.get(u, 0) + 1
          if low[v] >= dfn[u]:
            while st.pop() != v:
              continue
        else:
          low[u] = min(low[u], dfn[v])
      cut[u] = cut.get(u, 0) + 1

    mask_dfs(self)

    ##### Partition Computations into Groups
    pcnt, visited, group_export = [0], set(), dict()

    def compute_dfs(u, vid, is_leader):
      if u in visited:
        return
      if u.op_type != 'compute':
        return
      if is_leader:
        group_export[vid] = [u]
      else:
        group_export[vid].append(u)

      visited.add(u)
      for v in u.inputs:
        if cut.get(v, 0) > 1:
          pcnt[0] += 1
          compute_dfs(v, pcnt[0], True)
        else:
          compute_dfs(v, vid, False)

    compute_dfs(self, pcnt[0], True)

    compute_groups = []
    for _, members in sorted(group_export.items(), reverse=True):
      for x in members:
        multi_used = set()
        for y in x.inputs:
          if len(y.outputs) > 1:
            multi_used.add(y)
      compute_groups.append(([x for x in reversed(members)], multi_used))
    return compute_groups

  def get_data_parallel_config(self, **kwargs):
    visited = set()
    config = dict()

    def property_dfs(node):
      visited.add(id(node))
      for inp in node.inputs:
        if id(inp) not in visited:
          property_dfs(inp)
      config[node.name] = [-1, ""] if node.op_type == 'param' else [0, "BAR:0"]

    property_dfs(self)
    return Config.create(config, environ_config(kwargs))

  def serialize(self, **kwargs):
    node = self
    node.update_config(None, **kwargs)

    compute_groups = node.articulare_analyse()

    input_nodes, compute_nodes, config = [], [], {}
    visited = set()

    def property_dfs(node):
      visited.add(id(node))
      node_name = node.name
      for inp in node.inputs:
        if id(inp) not in visited:
          property_dfs(inp)
      if hasattr(node, 'config'):
        config[node_name] = getattr(node, 'config')
      if isinstance(node.data, dict):
        input_nodes.append(node)
      else:
        compute_nodes.append(node)
    property_dfs(node)

    return compute_groups, compute_nodes, input_nodes, config

  def compile(self, config, **kwargs):
    if not isinstance(config, dict):
      assert config.config['v'] == Config.VERSION, f"Unmatched configuration file version: expect {Config.VERSION}, got {config.config['v']}"
      for k in kwargs:
        config.config['kwargs'][k] = kwargs[k]
      kwargs = config.config['kwargs']
      config = config.config['b']

    run_mode = kwargs['run_mode']
    device_type = kwargs['device_type']
    total_nodes = kwargs['total_nodes']
    spmd_nodes = kwargs['spmd_nodes']
    assert total_nodes % spmd_nodes == 0, f"`total_nodes` must by evenly divided by `spmd_nodes`, got: {total_nodes} % {spmd_nodes} != 0"

    if True:
      _, compute_nodes, input_nodes, restricted_state = self.serialize(**kwargs)

      # Verify restricted_state & extra padding
      for node in compute_nodes + input_nodes:
        node_state = config[node.name][0]
        if restricted_state.get(node.name, node_state) != node_state:
          raise Exception(f"Unstatisfied sharding state requirements on node `{node.name}`")
        if node_state >= 0 and node.shape[node_state] % spmd_nodes != 0:
          raise Exception(f"Unstatisfied slicing chunks `{node.shape[node_state]} // {spmd_nodes}` on node `{node.name}`")

      # Construct Inputs
      input_list, param_list = [], []
      for node in input_nodes:
        shard_dim, _ = config[node.name]
        if node.op_type != 'param':
          input_list.append((node.name, session.backend.get_input_definition(node.name, node.shape, shard_dim, node.dtype, is_param=False)))
        else:
          param_list.append((node.name, session.backend.get_input_definition(node.name, node.shape, shard_dim, node.dtype, is_param=True)))

      def apply_communicate(item_name, comm_op):
        return re.sub(fr'\$', item_name, comm_op).strip()

      # Construct Computes
      graph_prog, temp_ids = [], 0
      for node in compute_nodes:
        output_dim, key = config[node.name]
        if ':' in key:
          key, rank = key.split(':')
          rank = int(rank)
        else:
          rank = None
        rule_func = solver.spmd_primitives_dict[key]
        conn_sol, conn_src = None, None
        try:
          valid_count = 0
          for rank, source_dims, connectors in rule_func(session, node, output_dim, spmd_nodes, rank):
            valid_count += 1
            assert valid_count <= 1, f"Ambiguous solution `{key}` for node with `{node.name}` at dimension {output_dim}"
            conn_sol, conn_src = connectors, source_dims
        except NotImplementedError:
          pass
        assert conn_sol is not None, f"No statisfied parallel pattern `{key}` applying on node `{node.name}`"

        graph_prog += [f'{node.name} = {node.fw_ops}',]
        for index in range(len(node.inputs)):
          input_item = node.inputs[index]
          item_name = input_item.name
          from_state = config[item_name][0]
          prim_state = conn_src[index]
          if from_state != prim_state:
            extra = {'output_shape': node.inputs[index].shape, 'is_param': node.inputs[index].op_type == 'param'}
            if from_state == -2 and prim_state >= 0:
              item_name = session.backend.link(item_name, -2, -1, **extra)
              item_name = session.backend.link(item_name, -1, prim_state, **extra)
            else:
              item_name = session.backend.link(item_name, from_state, prim_state, **extra)

          if index in conn_sol:
            item_name = apply_communicate(item_name, conn_sol[index]) or item_name

          if item_name != input_item.name:
              temp_ids = temp_ids + 1
              graph_prog[-1] = f'_temp{temp_ids} = {item_name}; ' + re.sub(fr'\b{input_item.name}\b', f'_temp{temp_ids}', graph_prog[-1])

        aggr_output = apply_communicate(node.name, conn_sol.get('', ''))
        if aggr_output:
          graph_prog += [f'{node.name} = {aggr_output}']

    depends, headers = set(), []
    def compute_dependencies(nodes):
        for node in nodes:
            if id(node) in depends:
                continue
            depends.add(id(node))
            for dep in node["depends"]:
                compute_dependencies(dep)
            headers.append(node["data"])

    for node in compute_nodes:
        compute_dependencies(node.depends)

    program_strings = session.backend.generate_framework_code(device_type, spmd_nodes, total_nodes // spmd_nodes, run_mode, self.name, headers, input_list, param_list, graph_prog)
    return Program(program_strings, kwargs)

def environ_config(kwargs):
  if 'spmd_nodes' not in kwargs:
    kwargs['spmd_nodes'] = kwargs['total_nodes']
  if 'device_type' not in kwargs:
    kwargs['device_type'] = os.environ.get('DEVICE', 'cuda')
  if 'run_mode' not in kwargs:
    kwargs['run_mode'] = os.environ.get('MODE', 'train')
  assert kwargs['total_nodes'] % kwargs['spmd_nodes'] == 0, "`total_nodes` must be exactly divided by `spmd_nodes`."
  return kwargs

def optimize(node, **kwargs):
  kwargs = environ_config(kwargs)

  if session.is_strict_fmt:
    node = Id(node, op_name='Builtin')
    node.config = 0

  compute_groups, compute_nodes, input_nodes, config = node.serialize(**kwargs)

  print('<< TUNE Graph >>\n')
  print('\n'.join([f'| {x.name} <- new_{x.op_type}() | {x.dtype}{x.shape} | {getattr(x, "config", None)} |' for x in input_nodes]))
  print('---------------------------------------------------')
  print('\n'.join([f'| {x.name} <- {", ".join([x.name for x in x.inputs])} | {x.dtype}{x.shape} | "{x.data}" | {getattr(x, "config", None)} |' for x in compute_nodes]))
  print('\n>> config = %s\n' % (json.dumps(config),))
  sys.stdout.flush()
  return kwargs, solver.solve_partition(session, compute_groups, input_nodes=input_nodes, split_pref=config, kwargs=kwargs)

class Config:
  VERSION = '0.1'

  @staticmethod
  def load_from_file(filename):
    if filename is not None and os.path.exists(filename):
      return Config(filename)
    return None

  @staticmethod
  def create(config, environ, timecost=0):
    return Config({'v': Config.VERSION, 't': timecost, 'b': config, 'kwargs': environ})

  def __init__(self, config):
    if isinstance(config, dict):
      self.set_config(config)
    elif isinstance(config, str):
      with open(config, 'r') as fp:
        config = json.load(fp)
      self.set_config(config)
    else:
      raise Exception('Unsupported config value: %s' % config)

  def set_config(self, config):
    if config['v'] != Config.VERSION:
      raise Exception('Incompatible config version: expect %s, got %s' % (Config.VERSION, config['v']))
    self.config = config

  def __str__(self):
    return json.dumps(self.config)

  def save(self, filepath):
    with open(filepath, 'w') as fp:
      json.dump(self.config, fp)

def Id(x, op_name=None):
  layout = ''.join([chr(ord('a') + i) for i in range(len(x.shape))])
  return Custom(f'{layout} = {layout}', f'{x}', op_name=op_name)

def Tensor(shape, dtype, is_param=False):
  inp = Custom({"shape": shape, "dtype": dtype, "is_param": is_param}, inputs=[])
  if not is_param and session.is_strict_fmt:
    config = getattr(inp, 'config', session.manual_config.get(inp.name, None))
    if config is not None:
      if inp.name in session.manual_config:
        session.manual_config.pop(inp.name)
      inp.config = 0
      inp = Id(inp, op_name="Builtin")
      inp.config = config
    else:
      inp.config = 0
      inp = Id(inp, op_name="Builtin")
  return inp
