# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import unittest
import sys, subprocess

import GPUtil

class HelloworldCaller():
    """A class for run tutel helloworld example with different arguments"""
    def run(self, top, dtype, num_local_experts, show_step_time=True):
        """Run helloworld example"""
        command = 'python3 tutel/examples/helloworld.py --top ' + str(top) + ' --dtype ' + dtype + ' --num_local_experts ' + str(num_local_experts)
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        losses = []
        while p.poll() is None:
            line = p.stdout.readline().decode("utf8").split()
            if len(line) > 5:
                if line[2] == 'loss':
                    if dtype == 'float32':
                        losses.append(round(float(line[4][:-1]), 3))
                    else:
                        losses.append(round(float(line[4][:-1]), 1))
                if line[0] == '[Summary]':
                    print('step time:', line[5])
        p.stdout.close()
        return losses

class TutelTestCase(unittest.TestCase):
    """A class for tutel test cases."""
    def setUp(self):
        """Hook method for setting up the test"""
        self.GPUtype = GPUtil.getGPUs()[0].name
        with open('tests/test_baseline.json') as f:
            self.data = json.load(f)
        for i in range(8):
            for j in range(len(self.data[i]['losses'])):
                if '32' in self.data[i]['dtype']:
                    self.data[i]['losses'][j] = round(float(self.data[i]['losses'][j]), 3)
                else:
                    self.data[i]['losses'][j] = round(float(self.data[i]['losses'][j]), 1)
        self.tutelCaller = HelloworldCaller()

    def test_top1_fp32_1_expert(self):
        """Test helloworld with top1 gate, float32 dtype and 1 expert(s)."""
        show_step_time = False
        for i in range(len(self.data[2]['step_time'])):
            if self.data[2]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[2]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(1, 'float32', 1), self.data[2]['losses'])

    def test_top1_fp32_2_experts(self):
        """Test helloworld with top1 gate, float32 dtype and 2 expert(s)."""
        show_step_time = False
        for i in range(len(self.data[3]['step_time'])):
            if self.data[3]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[3]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(1, 'float32', 2), self.data[3]['losses'])

    def test_top1_fp16_1_expert(self):
        """Test helloworld with top1 gate, float16 dtype and 1 expert(s)."""
        show_step_time = False
        for i in range(len(self.data[0]['step_time'])):
            if self.data[0]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[0]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(1, 'float16', 1)[0:2], self.data[0]['losses'][0:2])

    def test_top1_fp16_2_experts(self):
        """Test helloworld with top1 gate, float16 dtype and 2 expert(s)."""
        show_step_time = False
        for i in range(len(self.data[1]['step_time'])):
            if self.data[1]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[1]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(1, 'float16', 2)[0:2], self.data[1]['losses'][0:2])

    def test_top2_fp32_1_expert(self):
        """Test helloworld with top2 gate, float32 dtype and 1 expert(s)."""
        show_step_time = False
        for i in range(len(self.data[6]['step_time'])):
            if self.data[6]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[6]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(2, 'float32', 1), self.data[6]['losses'])

    def test_top2_fp32_2_experts(self):
        """Test helloworld with top2 gate, float32 dtype and 2 expert(s)."""
        show_step_time = False
        for i in range(len(self.data[7]['step_time'])):
            if self.data[7]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[7]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(2, 'float32', 2), self.data[7]['losses'])

    def test_top2_fp16_1_expert(self):
        """Test helloworld with top2 gate, float16 dtype and 1 expert(s)."""
        show_step_time = False
        for i in range(len(self.data[4]['step_time'])):
            if self.data[4]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[4]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(2, 'float16', 1)[0:2], self.data[4]['losses'][0:2])

    def test_top2_fp16_2_experts(self): 
        """Test helloworld with top2 gate, float16 dtype and 2 expert(s)."""
        show_step_time = False
        for i in range(len(self.data[5]['step_time'])):
            if self.data[5]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[5]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(2, 'float16', 2)[0:2], self.data[5]['losses'][0:2])
