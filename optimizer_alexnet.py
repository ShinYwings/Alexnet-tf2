# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend_config
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf

class lr_func(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate = "learning_rate", name=None):
        super(lr_func, self).__init__()
        
        self.init_lr = learning_rate

    def __call__(self, step):

        global_step_float = tf.cast(step, tf.float32)
        
        # 종료전 3배 감소
        if global_step_float == 127:
        
            return tf.divide(self.init_lr, 3.0)
        
        return self.init_lr

    def get_config(self):
        return {
            'ini_lr' : self.init_lr,
            'prev_loss' : self.prev_loss,
            'loss' : self.loss,
            'name' : self.name,
        }

class AlexSGD(tf.keras.optimizers.Optimizer):

    # Subclasses should set this to True unless they override `apply_gradients`
    # with a version that does not have the `experimental_aggregate_gradients`
    # argument.  Older versions of Keras did not have this argument so custom
    # optimizers may have overridden `apply_gradients` without the
    # `experimental_aggregate_gradients` argument. Keras only passes
    # `experimental_aggregate_gradients` if this attribute is True.
    # Note: This attribute will likely be removed in an upcoming release.
    _HAS_AGGREGATE_GRAD = True
    
    def __init__(self,
                learning_rate="learning_rate",
                momentum=0.9,
                weight_decay=0.0005,
                nesterov= False,
                name="AlexSGD",
                **kwargs):
                
        super(AlexSGD, self).__init__(name,**kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._weight_decay = False
        if isinstance(weight_decay, ops.Tensor) or callable(weight_decay) or weight_decay > 0:
          self._weight_decay = True
        if isinstance(weight_decay, (int, float)) and (weight_decay < 0 or weight_decay > 1):
          raise ValueError("`weight_decay` must be between [0, 1].")
        self._set_hyper("weight_decay", kwargs.get("weight_decay", weight_decay))
        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
          self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
          raise ValueError("`momentum` must be between [0, 1].")
        
        # self._set_hyper("_is_first", True)
        self._is_first = True
        # TODO 바꾼거
        self._set_hyper("momentum", kwargs.get("momentum", momentum))
        
        # Tensor versions of the constructor arguments, created in _prepare().
        # self._set_hyweight_decay = weight_decay
        
        self._set_hyper("v", 1)
    
    # TODO 바꾼거
    # @classmethod
    # def from_config(cls, config):
    #     custom_objects = {'lr_func': lr_func}
    #     return super(AlexSGD, cls).from_config(config, custom_objects=custom_objects)
    
    # def _prepare(self, var_list):
    #     self._weight_decay_t = ops.convert_to_tensor_v2(self._weight_decay, name="weight_decay")
    #     self._v_t = ops.convert_to_tensor_v2(self._v, name="v")
    
    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        for var in var_list:
            self.add_slot(var, 'v')
        for var in var_list:
            self.add_slot(var, 'weight_decay')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AlexSGD, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
            self._get_hyper("momentum", var_dtype))
        apply_state[(var_device, var_dtype)]["v"] = array_ops.identity(
            self._get_hyper("v", var_dtype))
        # apply_state["weight_decay"] = tf.constant(self._weight_decay, name='alex_weight_decay')
        apply_state[(var_device, var_dtype)]["weight_decay"] = array_ops.identity(
            self._get_hyper("weight_decay", var_dtype))
        

    # TODO 바꾼거
    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients['lr_t'], dict(apply_state=apply_state)

    # def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
    #     # grads, tvars = list(zip(*grads_and_vars))
    #     # (grads, _) = tf.clip_by_global_norm(grads, clip_norm=10)
    #     return super(AlexSGD, self).apply_gradients(grads_and_vars, name=name, experimental_aggregate_gradients=experimental_aggregate_gradients)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        momentum_var = self.get_slot(var, "momentum")
        
        # TODO 바꾼거
        # lr_t = self._decayed_lr(var_dtype)
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)

        v = self.get_slot(var, "v")
        
        weight_decay_var = self.get_slot(var, "weight_decay")
        # weight_decay_var = math_ops.cast(self._weight_decay, var.dtype.base_dtype)
        
        prev_var = var
        # v_var = math_ops.cast(self._v_t, var.dtype.base_dtype)
        # update rule
        # v_(i+1) = momentum * v_i - 0.0005 * lr * w_i - lr * grad
        # w_(i+1) = w_i + v_(i+1)
        # w_i = var
        # v_i = init 0

        if self._is_first:
            self._is_first = False
            # v = state_ops.assign(v , 1, use_locking=self._use_locking)
            # v_var = state_ops.assign(v, 1)

            # print("vvar is",v_var)
            left = math_ops.mul(momentum_var,v)
            center_1 = math_ops.mul(weight_decay_var, lr_t)
            center_2 = math_ops.mul(center_1, var)
            right = math_ops.mul(lr_t, grad)
            sub_1 = math_ops.subtract(left, center_2)
            v_t = state_ops.assign(v, math_ops.subtract(sub_1, right), use_locking=self._use_locking)
            var_update = state_ops.assign(var, var+v_t, use_locking=self._use_locking)
            
        else:
            # TODO : 마지막에 lr = lr/3  그리고 pv_var == var 일때 lr = learning_rate /10

            left = math_ops.mul(momentum_var,v)
            center_1 = math_ops.mul(weight_decay_var, lr_t)
            center_2 = math_ops.mul(center_1, var)
            right = math_ops.mul(lr_t, grad)
            sub_1 = math_ops.subtract(left, center_2)
            v_t = state_ops.assign(v, math_ops.subtract(sub_1, right) , use_locking=self._use_locking)
            var_update = state_ops.assign(var, var+v_t, use_locking=self._use_locking)
            tf.print("var_update is", var_update)
        
        # TODO 어떻게 loss랑 이전 loss가 같은지 확인하는 방법
        # if var_update == var:
        #     state_ops.assign(lr_t, tf.divide(lr_t,10))
            # coefficients['learning_rate'] = lr_t/10

        updates = [var_update, v_t]
        tf.print("var_update is",var_update)
        tf.print("v_t is ", v_t)

        # return super(AlexSGD, self)._resource_apply_dense(grad, var, **kwargs)
    
    def _resource_apply_sparse(self, grad, var, apply_state=None):
        raise NotImplementedError("Sparse gradient updates are not supported.")
        # var_device, var_dtype = var.device, var.dtype.base_dtype
        # coefficients = ((apply_state or {}).get((var_device, var_dtype))
        #                 or self._fallback_apply_state(var_device, var_dtype))
        # momentum_var = self.get_slot(var, "momentum")
        
        # # TODO 바꾼거
        # # lr_t = self._decayed_lr(var_dtype)
        # lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)

        # v = self.get_slot(var, "v")
        
        # weight_decay_var = math_ops.cast(self._weight_decay, var.dtype.base_dtype)
        
        # prev_var = var
        # # v_var = math_ops.cast(self._v_t, var.dtype.base_dtype)
        # # update rule
        # # v_(i+1) = momentum * v_i - 0.0005 * lr * w_i - lr * grad
        # # w_(i+1) = w_i + v_(i+1)
        # # w_i = var
        # # v_i = init 0

        # if self._is_first:
        #     self._is_first = False
        #     # v = state_ops.assign(v , 1, use_locking=self._use_locking)
        #     # v_var = state_ops.assign(v, 1)

        #     # print("vvar is",v_var)
        #     left = math_ops.mul(momentum_var,v)
        #     center_1 = math_ops.mul(weight_decay_var, lr_t)
        #     center_2 = math_ops.mul(center_1, var)
        #     right = math_ops.mul(lr_t, grad)
        #     sub_1 = math_ops.subtract(left, center_2)
        #     v_t = state_ops.assign(v, math_ops.subtract(sub_1, right), use_locking=self._use_locking)
        #     var_update = state_ops.assign(var, var+v_t, use_locking=self._use_locking)
            
        # else:
        #     # TODO : 마지막에 lr = lr/3  그리고 pv_var == var 일때 lr = learning_rate /10

        #     left = math_ops.mul(momentum_var,v)
        #     center_1 = math_ops.mul(weight_decay_var, lr_t)
        #     center_2 = math_ops.mul(center_1, var)
        #     right = math_ops.mul(lr_t, grad)
        #     sub_1 = math_ops.subtract(left, center_2)
        #     v_t = state_ops.assign(v, math_ops.subtract(sub_1, right) , use_locking=self._use_locking)
        #     var_update = state_ops.assign(var, var+v_t, use_locking=self._use_locking)
        #     tf.print("var_update is", var_update)
        
        # # TODO 어떻게 loss랑 이전 loss가 같은지 확인하는 방법
        # # if var_update == var:
        # #     state_ops.assign(lr_t, tf.divide(lr_t,10))
        #     # coefficients['learning_rate'] = lr_t/10

        # updates = [var_update, v_t]
        # print("var_update is",var_update)
        # tf.print("v_t is ", v_t)

        # return super(AlexSGD, self)._resource_apply_sparse(grad, var, **kwargs)

    def get_config(self):
      config = super(AlexSGD, self).get_config()
      config.update({
          "learning_rate": self._serialize_hyperparameter("learning_rate"),
          "decay": self._serialize_hyperparameter("decay"),
          "momentum": self._serialize_hyperparameter("momentum"),
          "weight_decay": self._weight_decay_t,
          "is_first": self._is_first,
          "v": self._serialize_hyperparameter("v"),
      })
      return config