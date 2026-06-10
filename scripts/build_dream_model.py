import os, sys, types, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR')
# The tfjs converter pulls in tensorflow_hub, which fails to import on TF 2.18
# (removed tf.estimator) + new protobuf. save_keras_model never uses hub, so we
# replace it with an empty stub module before importing tensorflowjs.
sys.modules["tensorflow_hub"] = types.ModuleType("tensorflow_hub")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras as keras            # Keras 2 API (the tfjs converter needs v2)
from tf_keras.layers import BatchNormalization
import tensorflowjs as tfjs

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "..", "assets", "models", "dream")

incp = keras.applications.InceptionV3(weights="imagenet", include_top=False)

# InceptionV3's BatchNormalization layers use scale=False (no gamma). tf.js's BN
# gradient crashes when gamma is absent, which kills the Deep Dream ascent. Clone
# the model with scale=True BN (gamma initialised to 1 = identical forward) so the
# gradient works in the browser.
def clone_fn(layer):
    if isinstance(layer, BatchNormalization) and not layer.scale:
        cfg = layer.get_config(); cfg["scale"] = True
        return BatchNormalization.from_config(cfg)
    return layer.__class__.from_config(layer.get_config())

cloned = keras.models.clone_model(incp, clone_function=clone_fn)
for l in cloned.layers:
    ow = incp.get_layer(l.name).get_weights()
    if isinstance(l, BatchNormalization) and len(l.get_weights()) == 4 and len(ow) == 3:
        l.set_weights([np.ones_like(ow[0]), ow[0], ow[1], ow[2]])   # gamma=1, beta, mean, var
    else:
        l.set_weights(ow)

# sanity: cloned forward must match the original
xt = tf.random.normal([1, 96, 96, 3])
d = float(tf.reduce_max(tf.abs(incp(xt) - cloned(xt))))
print("max forward diff after clone:", d, flush=True)

m = keras.Model(cloned.input, cloned.get_layer("mixed6").output, name="incp_mixed6")
print("params:", m.count_params(), "out shape:", m.output_shape, flush=True)
tfjs.converters.save_keras_model(m, out, quantization_dtype_map={"uint16": "*"})
print("converted to", out, flush=True)
