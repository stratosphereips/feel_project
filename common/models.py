import tensorflow as tf
from pathlib import Path

def get_ad_model(dropout=0.2):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(36)),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(20, activation='elu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(10, activation='elu'),
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


class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim=10, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(32, activation='elu')
        self.l2 = tf.keras.layers.Dense(20, activation='elu')
        self.l3 = tf.keras.layers.Dense(latent_dim, activation='sigmoid')
        self.drop = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        z = self.l3(x)

        return z


class Decoder(tf.keras.layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim=36, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(20, activation='elu')
        self.l2 = tf.keras.layers.Dense(32, activation='elu')
        self.l3 = tf.keras.layers.Dense(original_dim, activation='elu')
        self.drop = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        return self.l3(x)


class AutoEncoder(tf.keras.layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        # self.original_dim = 36
        self.encoder = Encoder(latent_dim=10)
        self.decoder = Decoder(36)
        self.margin = 2.0

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


def get_triplet_loss_model():
    inputs = tf.keras.Input(shape=(36))
    pos = tf.keras.Input(shape=(36))
    neg = tf.keras.Input(shape=(36))

    ae = AutoEncoder()
    outputs = ae(inputs, pos, neg)

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = tf.keras.Model([inputs, pos, neg], outputs)
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model


def get_classification_model(model_path: Path, n_classes=2, encoder_lr=0.001, classifier_lr=0.1,
              freeze_encoder_layers=False):
    encoder = _load_encoder_model(model_path)
    model = _add_classification_layers(encoder, n_classes, encoder_lr, classifier_lr, freeze_encoder_layers)

    return model


def _load_encoder_model(model_path: Path):
    ad_model = tf.keras.models.load_model(model_path)
    while len(ad_model.layers) > 6:
        ad_model.pop()
    ad_model.summary()
    return ad_model


def _add_classification_layers(model: tf.keras.Sequential, n_classes=2, encoder_lr=0.001, classifier_lr=0.1,
                              freeze_encoder_layers=False):
    if freeze_encoder_layers:
        for layer in model.layers[:]:
            layer.trainable = False

    encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_lr),
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=classifier_lr)

    model = model \
        .add(tf.keras.layers.Dense(6, activation='elu')) \
        .add(tf.keras.layers.Dense(n_classes if n_classes != 2 else 1, activation='elu'))

    multi_optimizer = tf.optimizers.MultiOptimizer(
        (encoder_optimizer, model[:-2]),
        (classifier_optimizer, model[-2:])
    )

    model.compile(optimezer=multi_optimizer, loss='bce' if n_classes == 2 else 'cce')

    return model
