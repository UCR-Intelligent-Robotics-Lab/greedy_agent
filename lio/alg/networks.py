import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# ── Keras-3 / TF-2.16+ compatibility shim ──────────────────────────────────
# tf.layers was removed in Keras 3. Re-expose the symbols via a lightweight
# wrapper so all downstream code that calls tf.layers.dense / batch_normalization
# / etc. continues to work without any algorithmic changes.
import types as _types

def _make_layers_shim():
    _mod = _types.ModuleType("tf.layers_shim")

    def dense(inputs, units, activation=None, use_bias=True,
              kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
              name=None, reuse=None):
        """Drop-in for tf.layers.dense with exact TF1 variable naming."""
        if kernel_initializer is None:
            kernel_initializer = tf.glorot_uniform_initializer()
        # Try static shape first
        last_dim = inputs.shape[-1]
        if hasattr(last_dim, "value"):
            input_dim = last_dim.value
        else:
            try:
                input_dim = int(last_dim)
            except (TypeError, ValueError):
                # Fallback to dynamic shape if static not available
                input_dim = tf.shape(inputs)[-1]
        with tf.variable_scope(name or "dense", reuse=reuse):
            kernel = tf.get_variable(
                "kernel",
                shape=[input_dim, units],
                initializer=kernel_initializer,
            )
            if use_bias:
                bias = tf.get_variable(
                    "bias",
                    shape=[units],
                    initializer=bias_initializer,
                )
            out = tf.matmul(inputs, kernel)
            if use_bias:
                out = tf.nn.bias_add(out, bias)
            if activation is not None:
                out = activation(out)
        return out

    def batch_normalization(inputs, training=False, name=None, reuse=None):
        with tf.variable_scope(name or "bn", reuse=reuse):
            return tf.keras.layers.BatchNormalization()(inputs, training=training)

    _mod.dense = dense
    _mod.batch_normalization = batch_normalization
    return _mod

tf.layers = _make_layers_shim()
# ── end shim ────────────────────────────────────────────────────────────────


def conv(t_input, scope, n_filters=6, k=(3, 3), s=(1, 1), data_format='NHWC'):

    if data_format == 'NHWC':
        channel_axis = 3
        strides = [1, s[0], s[1], 1]
        b_shape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_axis = 1
        strides = [1, 1, s[0], s[1]]
        b_shape = [1, n_filters, 1, 1]
    n_in = t_input.get_shape()[channel_axis].value
    w_shape = [k[0], k[1], n_in, n_filters]
    with tf.variable_scope(scope):
        w = tf.get_variable('w', w_shape)
        b = tf.get_variable('b', b_shape)
        out = b + tf.nn.conv2d(t_input, w, strides=strides, padding='SAME',
                               data_format=data_format)

    return out


def convnet(t_input, f=6, k=(3, 3), s=(1, 1)):
    h = tf.nn.relu(conv(t_input, 'c1', f, k, s))
    size = np.prod(h.get_shape().as_list()[1:])
    conv_flat = tf.reshape(h, [-1, size])

    return conv_flat


def actor(obs, n_actions, config, return_logits=False):
    conv_out = convnet(obs, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_probs')
    if return_logits:
        return out, probs
    else:
        return probs


def actor_mlp(obs, n_actions, config, return_logits=False):
    h1 = tf.layers.dense(inputs=obs, units=config.n_h1,
                         activation=tf.nn.relu,
                         use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu,
                         use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_softmax')
    if return_logits:
        return out, probs
    else:
        return probs


def actor_image_vec(obs_image, obs_vec, n_actions, config):
    """Actor network for inequity aversion agents.

    Args:
        obs_image: image part of observation
        obs_vec: agents observe vector of all agents' smoothed rewards
        n_actions: size of action space
        config: ConfigDict object
    """
    conv_out = convnet(obs_image, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='actor_h1')
    h1 = tf.concat([h1, obs_vec], axis=1)
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_probs')

    return probs


def reward(obs, a_others, config, n_recipients=1,
           output_nonlinearity=tf.nn.sigmoid):
    """Computes reward that this agent gives to (all) agents.

    Uses a convolutional net to process image obs.

    Args:
        obs: TF placeholder
        a_others: TF placeholder for observation of other agents' actions
        config: configDict
        n_recipients: number of output nodes
        output_nonlinearity: None or a TF function

    Returns: TF tensor
    """
    conv_out = convnet(obs, config.n_filters, config.kernel,
                       config.stride)
    conv_reduced = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                                   activation=tf.nn.relu, use_bias=True,
                                   name='reward_conv_reduced')
    concated = tf.concat([conv_reduced, a_others], axis=1)
    h2 = tf.layers.dense(inputs=concated, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='reward_h2')
    reward_out = tf.layers.dense(inputs=h2, units=n_recipients,
                                 activation=output_nonlinearity,
                                 use_bias=False, name='reward')
    return reward_out


def reward_mlp(obs, a_others, config, n_recipients=1,
               output_nonlinearity=tf.nn.sigmoid):
    """Computes reward that this agent gives to (all) agents.

    Args:
        obs: TF placeholder
        a_others: TF placeholder for observation of other agents' actions
        config: configDict
        n_recipients: number of output nodes
        output_nonlinearity: None or a TF function

    Returns: TF tensor
    """
    
    
    
    
    # Concatenate without the assertion
    concated = tf.concat([obs, a_others], axis=1)
    
    h1 = tf.layers.dense(inputs=concated, units=config.n_hr1,
                        activation=tf.nn.relu,
                        use_bias=True, name='reward_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_hr2,
                         activation=tf.nn.relu,
                         use_bias=True, name='reward_h2')
    reward_out = tf.layers.dense(inputs=h2, units=n_recipients,
                                 activation=output_nonlinearity,
                                 use_bias=False, name='reward')
    return reward_out


def vnet(obs, config):
    conv_out = convnet(obs, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='v_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='v_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None,
                          use_bias=True, name='v_out')

    return out


def vnet_mlp(obs, config):
    h1 = tf.layers.dense(inputs=obs, units=config.n_h1,
                         activation=tf.nn.relu,
                         use_bias=True, name='v_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu,
                         use_bias=True, name='v_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None,
                          use_bias=True, name='v_out')
    return out


def vnet_image_vec(obs_image, obs_vec, config):
    """Value function critic network for inequity aversion agents.

    Args:
        obs_image: image part of observation
        obs_vec: agents observe all agents' smoothed rewards
        config: ConfigDict object
    """
    conv_out = convnet(obs_image, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='v_h1')
    h1 = tf.concat([h1, obs_vec], axis=1)
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='v_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None,
                          use_bias=True, name='v_out')

    return out


