
import numpy as np
import tensorflow as tf
import shap


def explain_shap(model, x, features):
    baseline = np.zeros_like(x)

    def predict_masks(masks):
        masks = np.asarray(masks, dtype=np.float32)

        inputs = baseline[None, :, :] + (
            masks[:, None, :] * (x - baseline)[None, :, :]
        )

        return model.predict(inputs, verbose=0).flatten()

    explainer = shap.KernelExplainer(
        predict_masks,
        np.zeros((1, len(features))),
        feature_names=features
    )

    values = explainer.shap_values(
        np.ones((1, len(features))),
        nsamples=150,
        l1_reg=0,
        silent=True
    )

    values = np.asarray(values).reshape(-1)

    importance = sorted(
        zip(features, values),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    maximum = max(abs(value) for value in values) or 1

    return [
        {
            'name': name,
            'value': round(float(value), 3),
            'width': round(float(abs(value) / maximum * 100), 1),
            'direction': 'increase' if value >= 0 else 'decrease'
        }
        for name, value in importance
    ]


def explain_ig(model, x, steps=64):
    baseline = np.zeros_like(x, dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)

    path = baseline[None, :, :] + (
        alphas[:, None, None] * (x - baseline)[None, :, :]
    )

    with tf.GradientTape() as tape:
        tape.watch(path)
        predictions = model(path, training=False)

    gradients = tape.gradient(predictions, path)

    average = (
        gradients[:-1] + gradients[1:]
    ) / 2.0

    average = tf.reduce_mean(average, axis=0)
    attributions = (x - baseline) * average
    return attributions.numpy()


def make_heatmap(attributions, features):
    importance = np.abs(attributions).T
    maximum = importance.max() or 1

    rows = []

    for name, values in zip(features, importance):
        rows.append({
            'name': name,
            'cells': [
                {
                    'cycle': i + 1,
                    'strength': round(float(value / maximum), 3),
                    'value': round(float(value), 3)
                }
                for i, value in enumerate(values)
            ]
        })

    return rows
