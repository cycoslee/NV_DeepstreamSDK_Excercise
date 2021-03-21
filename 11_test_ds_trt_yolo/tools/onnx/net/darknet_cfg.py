import os

class Darknet_cfg():
    def __init__(self, config_file=""):
        self.blocks = [] 
        self.config_file = config_file

    def parse_cfg(self, config_file):
        if not os.path.isfile(config_file):
            print('WARN: Check parse_cfg caller call with config_file argument')
            print('WARN: Trying to check class variable(self.config_file) has the path')
            if not os.path.isfile(self.config_file):
                raise SystemExit('ERROR:Darknet tries to empty config file.')
            else:
                _config_file = self.config_file
        else:
            _config_file = config_file
        block = None
        fp = open(_config_file, 'r')
        line = fp.readline()
        while line != '':
            line = line.rstrip()
            if line == '' or line[0] == '#':
                line = fp.readline()
                continue
            elif line[0] == '[':
                if block:
                    self.blocks.append(block)
                block = dict()
                block['type'] = line.lstrip('[').rstrip(']')
                # set default value
                if block['type'] == 'convolutional':
                    block['batch_normalize'] = 0
            else:
                key, value = line.split('=')
                key = key.strip()
                if key == 'type':
                    key = '_type'
                value = value.strip()
                block[key] = value
            line = fp.readline()
        if block:
            self.blocks.append(block)
        fp.close()
        return self.blocks

    def print_cfg(self):
        if self.blocks == []:
            raise SystemExit('ERROR: Darknet has any blocks on it, please make sure that it has been parsed well.') 

        print('layer     filters    size              input                output');
        prev_width = 416
        prev_height = 416
        prev_filters = 3
        out_filters = []
        out_widths = []
        out_heights = []
        ind = -2
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                prev_width = int(block['width'])
                prev_height = int(block['height'])
                continue
            elif block['type'] == 'convolutional':
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                width = (prev_width + 2 * pad - kernel_size) // stride + 1
                height = (prev_height + 2 * pad - kernel_size) // stride + 1
                print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                    ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width,
                    height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                width = prev_width // stride
                height = prev_height // stride
                print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                    ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height,
                    filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'avgpool':
                width = 1
                height = 1
                print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (
                    ind, 'avg', prev_width, prev_height, prev_filters, prev_filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'softmax':
                print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'cost':
                print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                filters = stride * stride * prev_filters
                width = prev_width // stride
                height = prev_height // stride
                print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                    ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                filters = prev_filters
                width = prev_width * stride
                height = prev_height * stride
                print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                    ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    print('%5d %-6s %d' % (ind, 'route', layers[0]))
                    prev_width = out_widths[layers[0]]
                    prev_height = out_heights[layers[0]]
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                    prev_width = out_widths[layers[0]]
                    prev_height = out_heights[layers[0]]
                    assert (prev_width == out_widths[layers[1]])
                    assert (prev_height == out_heights[layers[1]])
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                elif len(layers) == 4:
                    print('%5d %-6s %d %d %d %d' % (ind, 'route', layers[0], layers[1], layers[2], layers[3]))
                    prev_width = out_widths[layers[0]]
                    prev_height = out_heights[layers[0]]
                    assert (prev_width == out_widths[layers[1]] == out_widths[layers[2]] == out_widths[layers[3]])
                    assert (prev_height == out_heights[layers[1]] == out_heights[layers[2]] == out_heights[layers[3]])
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + out_filters[
                        layers[3]]
                else:
                    print("route error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                            sys._getframe().f_code.co_name, sys._getframe().f_lineno))
    
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] in ['region', 'yolo']:
                print('%5d %-6s' % (ind, 'detection'))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'shortcut':
                from_id = int(block['from'])
                from_id = from_id if from_id > 0 else from_id + ind
                print('%5d %-6s %d' % (ind, 'shortcut', from_id))
                prev_width = out_widths[from_id]
                prev_height = out_heights[from_id]
                prev_filters = out_filters[from_id]
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'connected':
                filters = int(block['output'])
                print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters, filters))
                prev_filters = filters
                out_widths.append(1)
                out_heights.append(1)
                out_filters.append(prev_filters)
            else:
                print('unknown type %s' % (block['type']))

