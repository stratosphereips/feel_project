import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path


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

class FreezableModel(tf.keras.Model):
    def freeze(self):
        for layer in self.layers[:]:
            layer.trainable = False

    def unfreeze(self):
        for layer in self.layers[:]:
            layer.trainable = True


class Encoder(FreezableModel):
    def __init__(self, latent_dim=10, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(32, activation='elu')
        self.l2 = tf.keras.layers.Dense(20, activation='elu')
        self.l3 = tf.keras.layers.Dense(latent_dim, activation='sigmoid')
        self.drop = tf.keras.layers.Dropout(0.3)
        self.layers = [self.l1, self.drop, self.l2, self.drop, self.l3]

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


class Decoder(FreezableModel):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim=36, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(20, activation='elu')
        self.l2 = tf.keras.layers.Dense(32, activation='elu')
        self.l3 = tf.keras.layers.Dense(original_dim, activation='elu')
        self.drop = tf.keras.layers.Dropout(0.3)
        self.layers = [self.l1, self.drop, self.l2, self.drop, self.l3]

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
    def __init__(self, encoder_lr, decoder_lr, classifier_lr, disable_classifier=False):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim=10)
        self.decoder = Decoder(36)
        self.classifier = Classifier(10, 2, )

        multi_optimizer = tfa.optimizers.MultiOptimizer([
            (tf.keras.optimizers.Adam(learning_rate=encoder_lr), self.encoder.layers),
            (tf.keras.optimizers.Adam(learning_rate=decoder_lr), self.decoder.layers),
            (tf.keras.optimizers.Adam(learning_rate=classifier_lr), self.classifier.layers),
        ])





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
                             freeze_encoder_layers=False, dropout=0.3):
    encoder = _load_encoder_model(model_path)
    model = _add_classification_layers(encoder, n_classes, encoder_lr, classifier_lr, freeze_encoder_layers, dropout)

    return model


# def _load_encoder_model(model_path: Path):
#     ad_model = tf.keras.models.load_model(model_path)
#     while len(ad_model.layers) > 6:
#         ad_model.pop()
#     return ad_model

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


class Classifier(FreezableModel):
    def __init__(self, in_dim=10, n_classes=2, dropout=0.3):
        self.cl1 = tf.keras.layers.Dense(in_dim, name='DenseCls1', actication='sigmoid')
        self.drop = tf.keras.layers.Dropout(dropout, name='ClsDroupout')
        self.cl2 = tf.keras.layers.Dense(n_classes if n_classes != 2 else 1, name='DenseCls2', activation='sigmoid')
        self.layers = [self.cl1, self.drop, self.cl2]

    def call(self, X):
        X_prime = X
        for layer in self.layers:
            X_prime = layer(X_prime)
        return X_prime

    def freeze(self):
        for layer in self.layers[:]:
            layer.trainable = False

    def unfreeze(self):
        for layer in self.layers[:]:
            layer.trainable = True