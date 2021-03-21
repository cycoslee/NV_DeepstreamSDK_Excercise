from torch import from_numpy

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape));
    start = start + num_w
    return start

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape));
    start = start + num_w
    return start

def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start



