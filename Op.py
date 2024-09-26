import math
import numpy as np

class _Op():
    def __init__(self, name, op_type, attr, input_dims):
        self.name = name
        self.type = op_type
        self.attr = attr
        self.op_count = 0
        self.mac_count = 0
        self.wt_dims = [0]
        self.input_dims = input_dims
        self.output_dims = []
        self.n_inputs = 1
        self.n_outputs = 1
        
        self.map_attributes()
        self.calc_wt_dims()
        self.calc_output_dims()
        self.calc_ops()

    def calc_ops(self):
        for o in self.output_dims:
            self.op_count += int(np.prod(o))

    def calc_wt_dims(self):
        return

    def map_attributes(self):
        return

    def calc_output_dims(self):
        self.output_dims.append(self.input_dims[0])
    
class Convolution(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)

    def map_attributes(self):
        self.batch_size = self.input_dims[0][0] 
        self.input_channels = self.input_dims[0][1]
        self.in_x = self.input_dims[0][2]
        self.in_y = self.input_dims[0][3]

        for a in self.attr:
            attr_name = a.get('name')
            attr_value = a.get('value')
            if attr_name == 'dilation' and len(attr_value) == 0:
                attr_value = [1, 1]
            if attr_name in ['pad', 'dilation', 'kernel_size', 'stride'] and type(attr_value) == list and len(attr_value) != 2:
                attr_value.append(attr_value[0])

            setattr(self, attr_name, attr_value)
    
    def calc_wt_dims(self):        
        self.wt_dims = [self.kernel_size[0], self.kernel_size[1], self.input_channels, self.num_output]

    def calc_output_dims(self):
        if len(self.input_dims) > 1:
            print('Not handled!')

        self.out_x = math.floor((self.in_x + (2 * self.pad[0]) - (self.dilation[0] * (self.kernel_size[0] - 1))-1)/self.stride[0] + 1)
        self.out_y = math.floor((self.in_y + (2 * self.pad[1]) - (self.dilation[1] * (self.kernel_size[1] - 1))-1)/self.stride[1] + 1)
        self.output_dims.append([1, self.num_output, self.out_x, self.out_y])

    def calc_ops(self):
        self.mac_count = (self.input_channels * self.kernel_size[0] * self.kernel_size[1] * self.out_x * self.out_y) * self.num_output
        self.op_count = self.mac_count * 2

class BatchNorm(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)

    def calc_wt_dims(self):
        return
    
class Scale(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)

    def calc_wt_dims(self):
        return
    
class ReLU(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)

    def calc_wt_dims(self):
        return
    
class Eltwise(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)

    def calc_wt_dims(self):
        return
    
class Pooling(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)
    
    def map_attributes(self):
        self.batch_size = self.input_dims[0][0] 
        self.input_channels = self.input_dims[0][1]
        self.in_x = self.input_dims[0][2]
        self.in_y = self.input_dims[0][3]
        for a in self.attr:
            attr_name = a.get('name')
            attr_value = a.get('value')
            if attr_name in ['pad', 'dilation', 'kernel_size', 'stride']:
                if type(attr_value) == list and len(attr_value) != 2:
                    attr_value.append(attr_value[0])
                elif type(attr_value) == int:
                    attr_value = [attr_value, attr_value]
            setattr(self, attr_name, attr_value)

    def calc_output_dims(self):
        #output_height = (self.in_x - self.kernel_size[0] + 2 * padding) / self.stride[0] + 1
        #output_width = (self.in_y - self.kernel_size[1] + 2 * padding) / self.stride[1] + 1
        # TODO : where's the padding?
        output_height = math.floor((self.in_x - self.kernel_size[0]) / self.stride[0] + 1)
        output_width = math.floor((self.in_y - self.kernel_size[1]) / self.stride[1] + 1)
        self.output_dims.append([self.batch_size, self.input_channels, output_height, output_width])

    def calc_wt_dims(self):
        return
    
class InnerProduct(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)

    def map_attributes(self):
        self.batch_size = self.input_dims[0][0] 
        self.input_channels = self.input_dims[0][1]
        for a in self.attr:
            setattr(self, a.get('name'), a.get('value'))

    def calc_wt_dims(self):        
        self.wt_dims = [self.input_channels, self.num_output]

    def calc_output_dims(self):
        self.output_dims.append([1, self.num_output, self.input_dims[0][-2], self.input_dims[0][-1]])

    def calc_ops(self):
        self.mac_count = (self.input_channels * self.num_output)
        self.op_count = self.mac_count * 2

    
class Softmax(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)

    def calc_wt_dims(self):
        return
    
class Other(_Op):
    def __init__(self, name, op_type, attr, input_dims):
        super().__init__(name, op_type, attr, input_dims)

class Op():
    def __init__(self, name, op_type, attr, input_dims):
        if op_type in ["Convolution"]:
            self.instance = Convolution(name, op_type, attr, input_dims)
        elif op_type in ["BatchNorm"]:
            self.instance = BatchNorm(name, op_type, attr, input_dims)
        elif op_type in ["Scale"]:
            self.instance = Scale(name, op_type, attr, input_dims)
        elif op_type in ["ReLU"]:
            self.instance = ReLU(name, op_type, attr, input_dims)
        elif op_type in ["Eltwise"]:
            self.instance = Eltwise(name, op_type, attr, input_dims)
        elif op_type in ["Pooling"]:
            self.instance = Pooling(name, op_type, attr, input_dims)
        elif op_type in ["InnerProduct"]:
            self.instance = InnerProduct(name, op_type, attr, input_dims)
        elif op_type in ["Softmax"]:
            self.instance = Softmax(name, op_type, attr, input_dims)
        else:
            self.instance = Other(name, op_type, attr, input_dims)

    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.instance.__getattribute__(name)