from dataclasses import dataclass
from itertools import zip_longest
from typing import Optional, List

import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path
import numpy as np
from copy import deepcopy

from common.config import Config
from common.utils import deserialize, serialize
from sklearn.metrics import mean_squared_error, log_loss

def get_ad_model(dropout=0.2):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(36)),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(20, activation='elu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(10, activation='sigmoid'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(20, activation='elu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(36, activation='elu')
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def get_triplet_loss_model():
    inputs = tf.keras.Input(shape=(36))
    pos = tf.keras.Input(shape=(36))
    neg = tf.keras.Input(shape=(36))

    ae = AutoEncoder()
    outputs = ae(inputs, pos, neg)

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = tf.keras.Model([inputs, pos, neg], outputs)
    model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=True)

    return model


def get_classification_model(model_path: Path, n_classes=2, encoder_lr=0.001, classifier_lr=0.1,
                             freeze_encoder_layers=False, dropout=0.3):
    encoder = _load_encoder_model(model_path)
    model = _add_classification_layers(encoder, n_classes, encoder_lr, classifier_lr, freeze_encoder_layers, dropout)

    return model


class AutoEncoder(tf.keras.layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        # self.original_dim = 36
        self.encoder = Encoder(latent_dim=10)
        self.decoder = Decoder(36)
        self.margin = 1.0

    def call(self, inputs, positive, negative):
        z = self.encoder(inputs)
        p = self.encoder(positive)
        n = self.encoder(negative)
        reconstructed = self.decoder(z)

        # Add Triplet loss.
        # The triplet loss is defined as:
        # L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)

        ap_distance = tf.reduce_sum(tf.square(z - p))
        an_distance = tf.reduce_sum(tf.square(z - n))

        triplet_loss = ap_distance - an_distance
        triplet_loss = tf.maximum(triplet_loss + self.margin, 0.0)
        # triplet_loss = (ap_distance + tf.maximum(0.0, (10. - an_distance)/10.)) / 2.

        self.add_loss(triplet_loss)
        # self.add_metric(triplet_loss, name="triplet_loss", aggregation="mean")

        return reconstructed


class MultiHeadAutoEncoder(tf.keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        self.disable_classifier = config.model.disable_classifier
        latent_dim = config.model.latent_dim
        self.variational = config.model.variational
        self.encoder = Encoder(latent_dim=latent_dim)
        self.tracker = MetricsTracker()

        if self.variational:
            self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")
            self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")
            self.sampling = Sampling()

        self.decoder = Decoder(36)
        self.decoder.trainable = False

        self.classifier = Classifier(config.model.classifier_hidden, 2)
        self.classifier.trainable = not self.disable_classifier


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.model.learning_rate)

        self.loss = MultiHeadLoss(
            self.tracker,
            disable_classifier=config.model.disable_classifier,
            variational=config.model.variational,
            spheres={},
            disable_reconstruction=config.model.disable_reconstruction
        )

        self.proximal = config.model.proximal
        self.mu = config.model.mu
        self.prev_weights = self.get_weights()
        # self.optimizer = tfa.optimizers.MultiOptimizer([
        #     (tf.keras.optimizers.Adam(learning_rate=encoder_lr), self.encoder),
        #     (tf.keras.optimizers.Adam(learning_rate=decoder_lr), self.decoder),
        #     (tf.keras.optimizers.Adam(learning_rate=classifier_lr), self.classifier),
        # ])

    def call(self, inputs):
        X_true, y_true = inputs[:, :-1], inputs[:, -1]
        embedded = self.encoder(X_true)

        if self.variational:
            z_mean = self.z_mean(embedded)
            z_log_var = self.z_log_var(embedded)
            embedded = self.sampling([z_mean, z_log_var])

        reconstructed = self.decoder(embedded[y_true == 0])

        n_benign, dim = reconstructed.shape
        n_mal = y_true.shape[0] - n_benign

        ben_mask = (y_true == 0)
        mal_mask = (y_true == 1)
        ben_mask_diag = tf.linalg.tensor_diag(tf.cast(ben_mask, 'float32'))
        mal_mask_diag = tf.linalg.tensor_diag(tf.cast(mal_mask, 'float32'))

        ben_mask_matrix = tf.boolean_mask(ben_mask_diag, ben_mask, axis=0)
        mal_mask_matrix = tf.boolean_mask(mal_mask_diag, mal_mask, axis=0)

        reconstructed_all = tf.transpose(mal_mask_matrix) @ (tf.ones((n_mal, dim)) * -10)
        reconstructed_all += tf.transpose(ben_mask_matrix) @ reconstructed

        out = [reconstructed_all, embedded]
        if not self.disable_classifier:
            y_pred = self.classifier(embedded)
            out.append(y_pred)

        if self.variational:
            out += [z_mean, z_log_var]

        if self.proximal:
            prox_loss = 0
            for w, w_t in zip(self.get_weights(), self.prev_weights):
                if not w.any():
                    continue
                prox_loss += tf.norm(w - w_t)
            prox_loss *= self.mu
            self.tracker.proximal.update_state(prox_loss)
            self.add_loss(prox_loss)

        return tf.concat(out, axis=1)

    @property
    def metrics(self):
        return self.tracker.get_trackers()

    def compile(self):
        super().compile(optimizer=self.optimizer, loss=self.loss, run_eagerly=True)
        self.built = True
        self.predict(np.zeros((32, 36)))

    def predict(self, inputs):
        z = self.embed(inputs)
        reconstructed = self.decoder(z)
        y_pred = self.classifier(z)

        return tf.concat([reconstructed, y_pred], axis=1).numpy()

    def embed(self, inputs):
        embedded = self.encoder(inputs)
        if self.variational:
            z_mean = self.z_mean(embedded)
            z_log_var = self.z_log_var(embedded)
            embedded = self.sampling([z_mean, z_log_var])
        return embedded

    def set_spheres(self, spheres):
        self.loss.spheres = spheres

    def set_local_spheres(self, spheres):
        self.loss.local_spheres = spheres

    def get_weights(self):
        weights = super().get_weights()
        if self.disable_classifier:
            cls_layers = len(self.classifier.weights)
            for i, w in enumerate(weights[-cls_layers:]):
                weights[-cls_layers + i] = np.zeros_like(w)
        return weights

    def set_weights(self, weights):
        super().set_weights(weights)
        self.prev_weights = deepcopy(weights)

    def eval(self, X, y_true):
        predict = self.predict(X)
        X_pred, y_pred = predict[:, :-1], predict[:, -1]
        mse = np.mean(np.power(X - X_pred, 2), axis=1)
        return mse, y_pred, log_loss(y_true, y_pred)


class MetricsTracker:
    def __init__(self):
        self.total = tf.keras.metrics.Mean(name="total_loss")
        self.classification = tf.keras.metrics.Mean(name="class_loss")
        self.reconstruction = tf.keras.metrics.Mean(name="rec_loss")
        self.positive = tf.keras.metrics.Mean(name="+loss")
        self.negative = tf.keras.metrics.Mean(name="-loss")
        self.proximal = tf.keras.metrics.Mean(name='prox_loss')
        self.kl = tf.keras.metrics.Mean(name="kl_loss")

    def get_trackers(self) -> List['MetricsTracker']:
        trackers = [self.total, self.classification, self.reconstruction, self.proximal, self.positive, self.negative,
                    self.kl]
        return trackers

    def __add__(self, other: 'MetricsTracker'):
        self.total.merge_state([other.total])
        self.classification.merge_state([other.classification])
        self.reconstruction.merge_state([other.reconstruction])
        self.positive.merge_state([other.positive])
        self.negative.merge_state([other.negative])
        self.proximal.merge_state([other.proximal])
        self.kl.merge_state([other.kl])
        return self

    def serialize(self):
        return serialize([tf.keras.metrics.serialize(tracker) for tracker in self.get_trackers()])

    @staticmethod
    def deserialize(state):
        tracker = MetricsTracker()
        state = [tf.keras.metrics.deserialize(s) for s in deserialize(state)]
        tracker.total, tracker.classification, tracker.reconstruction,\
        tracker.proximal, tracker.positive, tracker.negative, tracker.kl = state
        return tracker


class MultiHeadLoss(tf.keras.losses.Loss):
    def __init__(
            self,
            tracker: MetricsTracker,
            dim=36,
            variational=False,
            disable_classifier=False,
            spheres=None,
            disable_reconstruction=False
    ):
        super().__init__()
        self.dim = dim
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mce = tf.keras.losses.MeanSquaredError()
        self.tracker = tracker
        self.variational = variational
        self.disable_classifier = disable_classifier
        self.disable_reconstruction = False
        self.spheres = spheres if spheres is not None else {}
        self.local_spheres = None
        self._epsilon = 1e-9

    def call(self, inputs_true, inputs_pred):
        """

        @param inputs_true: Expects the shape of the array be [X, y]; shape = (n_samples, dim + 1)
        @param inputs_pred: Expect the shape of array to be [X_reconstructed, X_embedded, y];
        shape = (n_samples, dim + embedding_dim + 1)
        @return:
        """
        total_loss = 0
        if self.variational:
            z_mean = inputs_pred[:, -4:-2]
            z_log_var = inputs_pred[:, -2:]
            inputs_pred = inputs_pred[:, :-4]
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            self.tracker.kl.update_state(kl_loss)
            total_loss += kl_loss

        label_true = inputs_true[:, -1]
        label_pred = inputs_pred[:, -1]

        embedded = inputs_pred[:, self.dim:] if self.disable_classifier else inputs_pred[:, self.dim:-1]
        #
        # if self.spheres:
        #     positive_loss = self.compute_positive_loss(embedded, label_true) / 1000
        #     negative_loss = self.compute_negative_loss(embedded, label_true)
        #     self.tracker.positive.update_state(positive_loss)
        #     self.tracker.negative.update_state(negative_loss)
        #     total_loss += positive_loss + negative_loss

        reconstructed_true = inputs_true[:, :self.dim]
        reconstructed_pred = inputs_pred[:, :self.dim]

        reconstructed_true_ben = tf.boolean_mask(reconstructed_true, label_true == 0, axis=0)
        reconstructed_pred_ben = tf.boolean_mask(reconstructed_pred, label_true == 0, axis=0)

        cross_entropy_loss = self.bce(label_true, label_pred) if not self.disable_classifier else 0.0
        if tf.math.is_nan(cross_entropy_loss):
            cross_entropy_loss = self._epsilon
        self.tracker.classification.update_state(cross_entropy_loss)
        total_loss += cross_entropy_loss

        reconstruction_loss = self.mce(reconstructed_true_ben, reconstructed_pred_ben)\
            if not self.disable_reconstruction else 0.0
        self.tracker.reconstruction.update_state(reconstruction_loss)

        total_loss += reconstruction_loss
        self.tracker.total.update_state(total_loss)

        return total_loss

    def compute_positive_loss(self, embedding, y):
        if not self.local_spheres:
            return 0
        loss_acc = 0
        # for client in self.spheres.values():
        for cls, (center, radius) in self.local_spheres.items():  # client.items():
            x = embedding[y == cls]
            safe_norm = tf.sqrt(
                tf.reduce_sum(tf.square(x - center), axis=1) + self._epsilon
            )
            dist = safe_norm  # tf.norm(x - center, ord='euclidean', axis=1)
            # dist = tf.clip_by_value(dist, 0, tf.float32.max)
            loss_acc += tf.math.reduce_sum(dist)
        return loss_acc

    def compute_negative_loss(self, embedding, y):
        loss_acc = 0
        for client in self.spheres.values():
            for cls, (center, radius) in client.items():
                x_neg = embedding[y != cls]

                safe_norm = tf.sqrt(
                    tf.reduce_sum(tf.square(x_neg - center), axis=1) + self._epsilon
                )
                dist = radius - safe_norm  # tf.norm(x_neg - center, ord='euclidean', axis=1)
                dist = tf.clip_by_value(dist, 0, tf.float32.max)
                # dist = tf.math.square(dist)

                loss_acc += tf.math.reduce_sum(dist)
        return loss_acc


class Classifier(tf.keras.layers.Layer):
    def __init__(self, in_dim=10, n_classes=2, dropout=0.3):
        super().__init__()
        self.cl1 = tf.keras.layers.Dense(in_dim, name='DenseCls1', activation='elu')
        self.drop = tf.keras.layers.Dropout(dropout, name='ClsDroupout')
        self.cl2 = tf.keras.layers.Dense(n_classes if n_classes != 2 else 1, name='DenseCls2')# , activation='sigmoid')

    def call(self, X):
        X_prime = X
        X_prime = self.cl1(X_prime)
        X_prime = self.drop(X_prime)
        X_prime = self.cl2(X_prime)
        return X_prime


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim=10, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(32, activation='elu')
        self.l2 = tf.keras.layers.Dense(18, activation='elu')
        self.l3 = tf.keras.layers.Dense(latent_dim, activation='sigmoid')
        self.drop = tf.keras.layers.Dropout(0.3)
        self.layer_list = [self.l1, self.drop, self.l2, self.drop, self.l3]

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        z = self.l3(x)

        return z

    def sequential(self):
        return tf.keras.Sequential([
            self.l1,
            self.drop,
            self.l2,
            self.drop,
            self.l3,
        ])


class Decoder(tf.keras.layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim=36, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(18, activation='elu')
        self.l2 = tf.keras.layers.Dense(32, activation='elu')
        self.l3 = tf.keras.layers.Dense(original_dim, activation='elu')
        self.drop = tf.keras.layers.Dropout(0.3)
        self.layer_list = [self.l1, self.drop, self.l2, self.drop, self.l3]

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        return self.l3(x)


def _load_encoder_model(model_path: Path):
    ae_model = tf.keras.models.load_model(model_path, custom_objects={'AutoEncoder': AutoEncoder, 'Encoder': Encoder,
                                                                      "Decoder": Decoder})
    encoder = ae_model.layers[3].encoder
    # while len(ad_model.layers) > 6:
    #     ad_model.pop()
    return encoder.sequential()


def _add_classification_layers(model: tf.keras.Sequential, n_classes=2, encoder_lr=0.001, classifier_lr=0.1,
                               freeze_encoder_layers=False, dropout=0.3):
    if freeze_encoder_layers:
        for layer in model.layers[:]:
            layer.trainable = False

    encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_lr)
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=classifier_lr)

    layers_before = len(model.layers)

    model.add(tf.keras.layers.Dense(6, name='DenseCls1', activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(dropout, name="DropoutCls1"))
    # model.add(tf.keras.layers.Dense(4, name='DenseCls2', activation='sigmoid'))
    # model.add(tf.keras.layers.Dropout(dropout, name="DropoutCls2"))
    model.add(tf.keras.layers.Dense(n_classes if n_classes != 2 else 1, name='DenseCls3', activation='sigmoid'))

    classifier_layers = len(model.layers) - layers_before

    multi_optimizer = tfa.optimizers.MultiOptimizer([
        (encoder_optimizer, model.layers[:-classifier_layers]),
        (classifier_optimizer, model.layers[-classifier_layers:])
    ])

    model.compile(optimizer=multi_optimizer,
                  loss='bce' if n_classes == 2
                  else 'cce')

    return model
