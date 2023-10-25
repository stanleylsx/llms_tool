# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : copy from https://github.com/airaria/TextPruner/blob/main/src/textpruner/utils.py
# @Email : gzlishouxian@gmail.com
# @File : print_parameters.py
# @Software: PyCharm
import torch


class LayerNode:
    def __init__(self, name, parent=None, value=None, fullname=None):
        self.name = name
        self.fullname = fullname
        self.value = None
        self.children_name = {}
        self.parent = parent

    def __contains__(self, key):
        return key in self.children_name

    def __getitem__(self, key):
        return self.children_name[key]

    def __setitem__(self, key, value):
        self.children_name[key] = value

    def update(self, value):
        if self.parent:
            if self.parent.value is None:
                self.parent.value = value
            else:
                if isinstance(value, (tuple, list)):
                    old_value = self.parent.value
                    new_value = [old_value[i] + value[i] for i in range(len(value))]
                    self.parent.value = new_value
                else:
                    self.parent.value += value
            if self.name.endswith('(shared)'):
                if self.parent.name.endswith('shared)'):
                    pass
                elif self.parent.value[0] == 0:
                    self.parent.name += '(shared)'
                else:
                    self.parent.name += '(partially shared)'

            self.parent.update(value)

    def format(self, level=0, total=None, indent='--', max_level=None, max_length=None):
        string = ''
        if total is None:
            total = self.value[0]
        if level == 0:
            max_length = self._max_name_length(indent, '  ', max_level=max_level) + 1
            string += '\n'
            string += f"{'LAYER NAME':<{max_length}}\t{'#PARAMS':>15}\t{'RATIO':>10}\t{'MEM(MB)':>8}\n"

        if max_level is not None and level == max_level:
            string += f"{indent + self.name + ':':<{max_length}}\t{self.value[0]:15,d}\t{self.value[0] / total:>10.2%}\t{self.value[1]:>8.2f}\n"
        else:
            if len(self.children_name) == 1:
                string += f"{indent + self.name:{max_length}}\n"
            else:
                string += f"{indent + self.name + ':':<{max_length}}\t{self.value[0]:15,d}\t{self.value[0] / total:>10.2%}\t{self.value[1]:>8.2f}\n"
            for child_name, child in self.children_name.items():
                string += child.format(level + 1, total,
                                       indent='  ' + indent, max_level=max_level, max_length=max_length)
        return string

    def _max_name_length(self, indent1='--', indent2='  ', level=0, max_level=None):
        length = len(self.name) + len(indent1) + level * len(indent2)
        if max_level is not None and level >= max_level:
            child_lengths = []
        else:
            child_lengths = [child._max_name_length(indent1, indent2, level=level + 1, max_level=max_level)
                             for child in self.children_name.values()]
        max_length = max(child_lengths + [length])
        return max_length


def summary(model, max_level):
    """
    Show the summary of model parameters.

    Args:
        model: the model to be inspected, can be a torch module or a state_dict.
        max_level: The max level to display. If ``max_level==None``, show all the levels.
    Returns:
        A formatted string.

    Example::

        print(textpruner.summay(model))

    """
    if isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model
    else:
        raise TypeError('model should be either torch.nn.Module or a dict')
    hash_set = set()
    model_node = LayerNode('model', fullname='model')
    current = model_node
    for key, value in state_dict.items():
        names = key.split('.')
        for i, name in enumerate(names):
            if name not in current:
                current[name] = LayerNode(name, parent=current, fullname='.'.join(names[:i + 1]))
            current = current[name]

        if (value.data_ptr()) in hash_set:
            current.value = [0, 0]
            current.name += '(shared)'
            current.fullname += '(shared)'
            current.update(current.value)
        else:
            hash_set.add(value.data_ptr())
            current.value = [value.numel(), value.numel() * value.element_size() / 1024 / 1024]
            current.update(current.value)

        current = model_node

    result = model_node.format(max_level=max_level)

    return result


def print_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}')
