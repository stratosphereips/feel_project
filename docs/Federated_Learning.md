# Federated Learning
Since the introduction of Federated Learning (FL), many variants have been
developed targeting specific use cases. The basic version of FL assumes a
central aggregator node, which communicates with a number of clients and
orchestrates the model training with them. In this case, the clients
hold the data, which usually have different distributions, but the features
observed in every client are the same, and the different clients usually hold
data describing different entities. In this document, we introduce some variants
of Federated Learning which differ in those basic assumptions.


## Cross-device and Cross-silo Federated Learning
The first applications of FL focused mainly on training on edge devices like
mobile phones or IoT devices. We refer to those approaches as cross-device
as the learning and data are present on the devices themselves. This is to
oppose a cross-silo approach, where the data is usually first siloed across
an organization onto a more powerful server, which then serves as a node
participating in the Federate Learning. This, for example, allows multiple
organizations to collectively train a model without sharing their data directly

![image](https://github.com/user-attachments/assets/20c17a29-fe6b-4590-bc04-9d462d8e5aed)


Figure: Classification of Federated Learning based on the type of participants.
In cross-device FL, the models are trained in the devices from which the data
originates. As opposed to cross-silo, where the participants are usually larger
organizations that have dedicated servers in which the data of the organization are
siloed. These servers usually provide better reliability and more computational
resources available for the training. The number of participants is generally
lower in the cross-silo setting.


Cross-device methods are often designed to handle thousands of participants
and are thus more difficult to coordinate and can have higher demands on
transmission efficiency. Furthermore, the edge devices participating in crossdevice learning often offer limited computational power or availability. For
instance, when training on mobile phones, conditions such as the charging
state, network connectivity, and the user’s activity must be considered. This is usually done by selection strategies, which would only accept
participating devices that fulfill defined conditions. It is also assumed that
some devices might become unavailable during the training process.
In the cross-silo scenario, the data from an organization are usually concentrated on a single machine within an organization. This and a number
of participants as low as two, allow for stronger assumptions on availability,
computational resources, and network connectivity.
Sometimes, the cross-silo approach is employed when organizations share
the same entities in their datasets but observe different features describing
them, but for privacy reasons, they can’t share data directly. This can be
the case, for example, when hospitals and laboratories conduct medical tests
and have data about the same individuals. Due to the high privacy concerns

related to medical records, privacy-preserving FL can be the only option to
train a model using both datasets. This approach is called Vertical Federated
Learning and is further discussed in the next section.

## Horizontal and Vertical Federated Learning
One of the key classifications of Federated Learning is how the samples and
features are partitioned among the clients participating in FL. When the
features in the datasets are the same, but the samples come from different
(or partially overlapping) sets of entities, it is called Horizontal Federated
Learning (HFL). The opposite case, when clients share different features from
the same set of entities, it is called Vertical Federated Learning (VFL).
HFL is applicable in cases where similar behaviors can be observed in
a number of clients. For example, it can be used to improve wake word
detection models used by AI assistants. In this case, each device collects the
same audio features, but each device contains only samples generated by a
single user. In this setup, a CNN is able to learn to detect key phrases with
high accuracy without the data leaving the edge devices.
The concept of the vertical split of the data is more applicable in the
cross-silo FL. It assumes that all participants are in possession of data from
the same ID space but different feature spaces. This has direct effects on
the architecture of the models. In HFL, all clients generally have the same
models, which are being locally updated by them and then synced through
a central coordinator. While in VFL, the clients usually only maintain part
of the model, and both training and inference have to be done in a more
coordinated manner.
In the case of neural networks, each client maintains a part of the whole
model. During training, it inputs its set of features through its model and
sends the output to a participant in possession of the labels. The responsibility
of the label owner is to produce an output of the whole model and send the
backward messages to all participants so that they can update their models.
One of the pre-requisites for this process of training is the establishment
of a common set of IDs of samples which further adds to the coordination
complexity


## Federated Optimization

In this subsection, we will introduce the main methods used for federated
optimization, which refers to the optimization algorithms used in Federated
Learning. The focus of this subsection is on introducing the main methods
used for training models in FL, which are mostly extensions of Stochastic
Gradient Descent (SGD) and can be used to train a variety of models, including

Neural Networks, Linear Regression, Support Vector Machine,
Gradient Boosting Trees, and Random Forests.
Federated Learning SGD can be implemented with the Federated Averaging
algorithm (FedAvg) (described in the [Federated Optimization section](https://github.com/stratosphereips/feel_project/blob/main/docs/Federated_Learning.md#federated-optimization)). Since its introduction, new
methods were proposed to extend and improve upon it in order to achieve
better stability, and performance. This includes both the performance of the
model and the computational efficiency of the training process



## Federated Averaging

FedAvg was proposed by McMahan et al., and it can be viewed as an
extension of Stochastic Gradient Descent and is described in Algorithm 1. The
training of the models is done in rounds, coordinated by a central aggregator
server. The server initializes a model w0, which then broadcasts to all clients
who take part in the training. The clients then apply a local SGD using their
data Dk for a number of epochs - E. After the clients finish their rounds,
they send the adjusted weights wk
t of their models back to the central server,
which aggregates the weights by computing their average. This results in a
new global model wt, which is broadcasted to the clients in the next round
of training.
The main contribution of the FedAvg compared to previous algorithms
used distributed learning is the introduction of the multiple local epochs.
This allows for more communication-efficient learning as the model parameters
are transmitted less often. However, higher values of E also lead to more
divergence between the clients’ models in the case of non-IID data.


## Adaptive Optimization

Extensions of FedAvg were developed to allow for better control of the
training process. One of those is FedOpt proposed by Reddi et al.,
which generalizes the "vanilla" FedAvg and enables the use of momentumbased and adaptive optimization techniques. Algorithm 2 shows the federated variant of the ADAM optimizer, which utilizes these techniques together with separate learning rates for the client and server procedures. Similarly, as in
traditional machine learning, the adaptive methods in a federated setting tend
to have better convergence characteristics and necessitate less hyperparameter
tuning.
It is important to note that in some cases, the clients may be unable to
maintain a persistent state between the rounds. For example, when the client
population is large, and the participants in each iteration are sampled. In
this case, many clients may only be used once, and the local learning rates
can not be applied.

![image](https://github.com/user-attachments/assets/d61711e7-db8b-42b6-a31c-f9d19350ce13)


## Regularization

In FL, clients often perform multiple local updates before sharing them with
the aggregator server in order to save on communication overhead. This often
leads to client drift where the models of the individual clients diverge. This
can be mitigated by using regularization, which can be viewed as a term
added to the loss function, which penalizes the drifting of the local model
from the global one.
By choosing the function ψ(w, wg), a distance function of the local model
w, and the latest global model wg, we can ensure that the local models will
not diverge significantly. Examples of the regularization functions can be
found in the FedProx, which introduces the proximal term
![image](https://github.com/user-attachments/assets/cc161a78-2139-474a-8022-84f26f9209b3)


where µ is a hyper-parameter of the method. The proximal term is added to
the loss function as a penalty.

![image](https://github.com/user-attachments/assets/1039bc13-5fb0-4f10-a7eb-7f4e0898b67f)


## Model Weighting


During the model aggregation step on the central server, a weighting scheme
is often employed. For example, in the FedAvg algorithm, the global model
can be obtained as a weighted average of the clients’ contributions. Weighting
is most often employed in cases of data or performance heterogeneity, as
described in the [Challenges of Federated Learning section](https://github.com/stratosphereips/feel_project/blob/main/docs/Federated_Learning.md#challenges-of-federated-learning). Its main purpose is usually to mitigate the tendency
of the model to favor clients with higher volumes of training data or available
resources. However, employing some weighting scheme may require the clients
to share information about their data which might bring technical and privacy
concerns. For example, when weighting is based on the number of samples
used for training the model, the clients inevitably leak the amount of data
they contain.


## Challenges of Federated Learning

Federated Learning is a collaborative effort to extract knowledge from a set of
clients while ensuring their data privacy. As this process is usually distributed,
it shares some challenges with Distributed Learning while also introducing
some unique ones. In this document, we will discuss some of those challenges
and how they are usually addressed.


## Data Heterogeneity

One of the most discussed and wildly studied challenges in Federated Learning
is that the data across clients is often not identically and independently
distributed (non-IID). This usually arises when each device contains data
generated by a single user or a small set of them. It means that it is more
common in the cross-device setting, but organizations in cross-silo settings
can also each have uniquely skewed data.
The non-IID property of the data can manifest itself in a multitude of
ways. The class balance can be different across clients, some labels being
overrepresented on some sets of clients while non-present on others at all.
The quality of labels can also vary across clients, and it is possible that the
same features can have different labels across devices.
The non-IID data was presented as a key challenge in Federated Learning
since its introduction, and it is often viewed as a requirement for realistic
FL datasets.
Non-IID data is not unique to FL, and it is a challenge with which many ML
models struggle. Robust statistics is an area of research focusing on methods
applicable to data coming from a wide range of distributions. Using some
robust techniques can help with FL on non-IID data. Examples of this can
be extensions of FedAvg (see the [Federated Optimization section](https://github.com/stratosphereips/feel_project/blob/main/docs/Federated_Learning.md#federated-optimization)), which use a median or trimmed
mean instead of a simple mean for weight aggregation. These robust
methods offer resiliency against outliers but at the cost of losing information,
leading to slower convergence and worse performance.
A different approach is to relax the requirement of training a single global
model. As the issues of non-IID data arise from the heterogeneity of the
client’s data, a possible solution might be to personalize the models for each
client. A common approach is to use a method common in transfer learning -
local fine-tuning. In it, a shared FL model is first acquired, which
is then fine-tuned on the client’s own data to achieve personalization. Work
has been done relating FL to Meta Learning when viewing clients as
heterogeneous tasks. This allows for the use of Model Agnostic Meta Learning
(MAML), which strives for general models which can be quickly adapted
for specific tasks.

Device Heterogeneity
In many practical FL deployments, there exist significant differences among
the clients. We discussed the differences in their data in the previous section,
but apart from those, the clients can also vary in the available performance
and communication resources as well as in the amount of data they can
collect.
The practical challenge of training a federated model is that as the individual
clients differ in their performance, it may take longer for some of them to
finish iterating a batch of training data. This leads to some of the clients
sitting idle and the wall-clock time of the training increasing.
A solution is to set a fixed time limit and allow the clients who would
have finished sooner to continue training locally and omit the contributions
of the struggling clients. Another option is to use asynchronous aggregation
instead of conducting the training in distinct rounds. This allows the clients
to share their contributions with the aggregating server as often as they are
capable.
However, both of those approaches introduce bias favoring clients that are
capable of finishing more batches. To mitigate this, a weighing schema is
usually employed in the aggregation, decreasing the weights of contributions
produced by the more active client. A similar schema is also used to
promote contributions from clients with fewer training samples available, to
decrease the bias from the data volume heterogeneity.

Communication Efficiency
When deploying FL on IoT devices or mobile phones, communication overhead
can become a significant obstacle. The main sources of bandwidth consumption is the aggregation server sharing the global model with the clients and
the clients sending their gradients back to it. Some common techniques
used to decrease the model size can be applicable here, e.g., sparsification or
quantization.
Gradient compression is a popular technique that reduces the size of the
gradients by only preserving the most important components and sparsifying
them.
Another approach is to employ a Multi-Epoch aggregation, where in each
round the clients train their local models for multiple epochs, and only after
that they share their updates with the aggregator. This decreases the
total number of communication rounds but also allows the local models to
drift from each other.
These techniques usually represent a trade-off between the communication
efficiency and the convergence time of the model and its accuracy


# Methodology
## Proposed Solution
We use horizontal cross-device federated learning for detecting malicious
activity in encrypted TLS network traffic. Cross-device in this context means,
that the clients represent edge computers, monitoring and capturing their
traffic. It is horizontal because the clients observe the same set of features,
produced by different entities. The federated approach allows to distributively
train a model using the client’s observations, without having direct access to
the data. This enables us to protect the privacy of the data, while still being
able to learn from it. In addition, each client also benefits from cooperative
training, as they use a global detection model that is averaged from all model
updates sent by all the clients. The global model, therefore, had access to a
larger and more diverse set of data coming from all clients, possibly leading
to better performance and generalization, compared to a model trained only
with each client’s local data.

## Solution Architecture
Our federated learning system consists of a central aggregator and multiple
clients. The aggregator is a dedicated machine that initializes and coordinates
the training process. There are ten clients in our system, representing the
edge computers, each containing their data in the form of processed feature
vectors used for training the detection models. The data spans multiple
days, and each day is treated as a separate training process. The clients
split each day’s data into training and validation data using an 80/20 split.
The clients use the data of the next day as testing data for the current day.
This is functionally equivalent to training the model on yesterday’s data and
evaluating it on today’s data as it is coming in.
At the start of the training, every client needs to adjust its features to a
common range. For this purpose, they each fit a MinMax scaler, which finds

## Diagram of the training process. Each day is treated as a separate
training process with multiple federated training rounds. The aggregator coordinates the training process, distributes the global models, and creates new
models using updates it receives from the clients. The clients distributively
train the models using their local data. In our work, there are up to ten clients
participating in the training.

the minimum and maximum values of every feature in the client’s training data.
Those extreme values are then shared with the aggregator, which combines
them to produce a global MinMax scaler that is then distributed back to the
clients. We choose the MinMax scaler, as it can be easily implemented in the
federated setting. This scaler traditionally scales the values to 0-1 range. It
is possible that some of the values transformed by the scaler may lay outside
the fitted range of the scaler, as the scaler fit on the training data is also used
to transform the validation and testing data. In this case, the values will be
scaled outside of the 0-1 range proportionally to the values observed in the
training data. This does not create issues for our work, as the used features
are computed on 1-hour windows and are mostly consistent.
The training itself is initiated by the clients receiving an initial global
model from the aggregator. The clients then train this model for a series
of rounds. A round consists of the clients training the models locally for
a number of epochs specified by the aggregator. After the local training,
each client reports how the model’s weights changed to the aggregator. In
turn, the clients receive an updated aggregated global model, which they
use in the next round. For the purposes of the experimental evaluation, the
clients also evaluate the new aggregated global model on their testing data,
which consists of the benign and malicious from the next day. They compute
relevant metrics on this dataset and report them back to the aggregator.
After which the next round of training starts.
The number of rounds and local epochs depends on the complexity of the
models and the number of training iterations it needs to converge. Increasing
the number of local epochs means that the model can be trained for the same
number of epochs while decreasing the number of rounds. This effectively
lowers the communication overhead. However, more local iterations may also


lead to a higher risk of divergence of the models, as the clients receive the
global models less often. We discuss the exact number of rounds and local
epochs in the following sections describing the individual approaches.
The aggregator combines metrics using a weighted average with the number
of clients’ samples as weights. It also produces a global model after each
round using a process described in the [Learning Algorithm section](https://github.com/stratosphereips/feel_project/blob/main/docs/Federated_Learning.md#learning-algorithm). At the end of the day’s
training process, the aggregator selects the best-performing model using an
aggregated validation loss of each model. This model is used as the initial
model for the next day. We assume that when the model is reused, it can
be trained for fewer rounds as it already possesses some domain knowledge.
This can save on both computational and communication resources as well as
enable the model to preserve previous knowledge.


## Unsupervised Approach
For unsupervised learning, we use a Variational Autoencoder (VAE) to detect
anomalies in the network traffic. One reason for using unsupervised learning,
in this case, is that it might be difficult to obtain high-quality labels for
malware traffic, which are needed for supervised learning approaches. Although our dataset is labeled, previous works have reported that unsupervised
learning can be effective for detecting anomalies in network traffic data.
The anomaly detection model consists of an encoder and a decoder. The
encoder of the model embeds the inputs into a 10-dimensional latent space.
It fits multivariate normal distributions (with a diagonal covariance matrix)
from the data to generate the embedded samples. The decoder then attempts
to reconstruct the input vector from its compressed representation. The
architecture of the model is shown in Figure 4.2a
The model is trained using a combined loss function consisting of the reconstruction loss Lmse (Mean Square Error of the input and output vectors), and
the regularization Kullback-Leibler loss (KL), which penalizes the difference
between the learned distribution and the standard normal distribution. The
use of this penalty function was introduced together with the VAE, and
ensures that the learned distributions do not diverge from each other and
produce a generalizing embedding. The loss function for each sample can
be represented as:


For detection, the reconstruction loss is used as an anomalous likelihood.
Each client derives an anomaly threshold based on its validation data. The

![image](https://github.com/user-attachments/assets/2f0d0d51-ac72-4b86-ab7d-ace76adddc7b)


Figure: Architecture of the Neural Network models used in this work. The
Classifier-only model is derived from the Multi-Head model by removing its
reconstruction head.

reconstruction error on the normal data is often used for deriving the anomaly
threshold. The clients compute reconstruction errors for every sample
in their normal validation dataset. From those values, they found a threshold
that classifies 1 % of their validation data as malicious. We use this approach
in order to provide robustness to outliers in the benign validation data. If we
would instead choose the maximum value on the validation dataset, it could
select some non-malicious outlier of the dataset. Such a threshold would then
result in more malware being missed. The specific value of 1% was chosen
using an expert heuristic.

After computing individual thresholds, the clients send them back to the
aggregator, which averages them and weights them by the ratio of the training
data in each client to produce a global threshold. On the first day, when the
model is trained from scratch, ten training rounds are used, and five when
the model from the previous day is reused. Clients train the model locally for
one epoch in the first two rounds and for two epochs in the remaining rounds.
While more local rounds are more communication efficient, as discussed
in the [Challenges of Federated Learning section](https://github.com/stratosphereips/feel_project/blob/main/docs/Federated_Learning.md#challenges-of-federated-learning), it may also lead to divergence in the individual client’s
updates. When using the momentum-based methods, the largest steps in the
parameter space are generally made in the first few epochs. We hope
that by aggregating after each epoch in the first two rounds, the global model
manages to converge into a state from which it reliably converges with less
frequent aggregations.


## Supervised Approach
For supervised learning, we use two types of models, a Multi-Head model
and a Classifier-only model. They are shown in Figure 4.2b and Figure 4.2c,
respectively. The Multi-Head model is derived from a regular autoencoder by
adding a classification head after the encoder while also keeping the decoder
part. We hope that by keeping the autoencoder components, the clients with
only benign samples can contribute to the learning process by improving the
embedding space. The model is trained to distinguish malicious traffic from
benign; malicious being the positive class. Only benign samples are passed
to the decoder part of the network so that the network does not learn to
reconstruct the malicious samples well.

The model is trained on a combined classification - Binary cross entropy
(LBNE) - and reconstruction loss - Mean Square Error (LMSE).


The Classifier-only model was created to evaluate if the decoder part of
the Multi-Head model brings any benefits. Its structure is identical to the
Multi-Head model, but with the reconstruction part of the network missing.
This effectively turns it into a regular classification model and is trained using
only the Binary cross entropy.
The supervised models are trained for 75 rounds on the first day when
the models are trained from scratch. On the following days, when the model
from the previous day is reused, 25 training rounds are used. Clients train
the model locally for one epoch every round.

## Malware Vaccine
The motivation behind the Multi-Head model is to allow all clients to participate in supervised training, even if they do not have any malware data.
However, it was observed that having clients participating without positive
samples makes the supervised federated model unable to converge. To address
this, we decided to send each client a set of malicious feature vectors, which
we call a "vaccine", to help with the convergence. Traditionally in the security
field, vaccines are a harmless part of the malware that is injected into the
host machines to prevent infections. Our vaccines differ in that they are
not a passive mechanism but an aid in the learning process and a way to
tackle data heterogeneity. The vaccines are
comprised of only numerical values and, as such, do not pose any risk to the
clients. In our setting, the central aggregator is responsible for gathering and
distributing this set of data to the clients. This approach could be achieved in
real deployments by using samples of malicious network traffic from publicly
available datasets. In order to mitigate the convergence issues, the "vaccine"
dataset has to be sufficiently large (more than 70 feature vectors)

## Assumptions and Limitations
In this subsection, we discuss the assumptions and limitations of our work.
. Supervised learning assumption: In the supervised setting, we
assume that the clients have the capability to label the data locally. This
is a relatively strong assumption in real deployments, but our work aims
at developing methods that would only require this from a subset of the
participants.
. Unsupervised learning assumption: For the unsupervised setting,
we assume that there are no malicious samples in the benign dataset.
While we are confident that this assumption holds in this work, as the
dataset was created using expert knowledge, it may be challenging to
assure in future work or in real-world deployments.
. Client trust: We also assume that the clients who connect are not
malicious and try to damage the training process or extract knowledge
from other clients.
. Client availability: One limitation of our work is that we do not handle
cases where some clients drop out or are unavailable during the training
process. Although this is quite common in real-world settings, we believe
that with the limited size of the dataset, we would not be able to evaluate
this well.

Overall, our work has several assumptions and limitations that should be
considered when interpreting the results and implications. These limitations
do not invalidate our findings, but they should be taken into account when
considering the generalizability and applicability of our approach.

## Learning Algorithm
To train our models, we use a combination of FedAdam and FedProx
algorithms. SGD with FedProx regularization is used to train the models in
the clients. FedProx adds a term to the loss function penalizing the clients’
divergence from the last received global model. This mitigates divergence in
case of statistical differences between the clients’ datasets. This client-side
regularization is important when training with a small batch size (making
a larger number of local steps) or when training locally for multiple epochs
before sending the weight updates to the aggregator.
On the server side, the clients’ contributions are aggregated using a weighted
average based on the amount of clients’ training data. Using this aggregate,
a new update to the global model is created using the FedAdam algorithm
provided in the flower framework. It is a federated variant of the Adam
optimizer, and as such, it uses momentum when aggregating the client
updates providing better convergence on the heterogenic data. These learning
algorithms are described in more detail in the [Federated Optimization section](https://github.com/stratosphereips/feel_project/blob/main/docs/Federated_Learning.md#federated-optimization)

## Implementation
We chose to use a Python-based open-source federated learning framework
called Flower to implement our methods. Flower is a versatile framework
that provides extendable implementations of both the server (aggregator)
and the clients. It is designed to handle the communication between the
aggregator and clients, enabling us to focus on developing the methods. In
addition, Flower includes implementations of some of the common algorithms
for aggregating client updates.
In the Flower framework the user is responsible for implementing the
functionality, such as loading the local dataset, orchestrating the local fitting,
and computing the metrics. To train the models in the clients, we have used
TensorFlow in combination with Keras. On the server side, the
aggregation of metrics must be implemented, as well as the initialization of
the model and setting of training parameters. Flower allows for the sending of
serializable data structures, which can be useful for exchanging configurations
or other information between aggregator and clients. The serializable data
has been used to distribute the vaccines from the aggregator to the clients.



The code for this work can be found in this repository. This repository contains the implementation of the described methodology, including the code
for orchestrating and running experiments and analyzing their results. It also
includes the preprocessing of raw data into hourly feature vectors used by
the models. The feature extraction is based on an in part reused from the
work done by František Střasák.


## Experiment Setup
This section describes how the proposed methods are evaluated, how they
are compared, and how the metrics are collected.
For the purposes of this work, one experiment is a set of conditions and
parameters which are evaluated. Each experiment consists of ten federated
runs with identical parameters, differing only in the random seed used for
initializing the model and splitting the local datasets for training and validation. Each run in an experiment performs a federated run, a local run, and a
centralized run; all using the same parameters and random seed.
On each run, the models are trained for a total of four days, producing a
global detection model each day. These models are then evaluated on the
next day’s data resulting in four sets of evaluation metrics from each of the
runs. Only four days are evaluated because the dataset has five days and the
last day can not be evaluated since there is no next day.


## Federated Training Process
The dataset on which our solution is evaluated has network traffic from
five consecutive days for ten distinct clients. Only five clients have labeled
malicious traffic which can be used for supervised training, the rest five clients
only have benign traffic.
On each day a federated training process is done. The diagram in Figure 4.1
illustrates the federated training process. On the first day of training, the
model is initialized randomly and trained from scratch on each client’s data,
while on the following days (days 2, 3 and 4), the previous day’s model is
reused. On a given day, the clients train using data from that day, split into
training and validation data (using an 80/20 split). The validation dataset is
used for selecting the best-performing global model (based on an aggregated
validation loss). The testing is done on the next day’s traffic, which is
functionally equivalent to using the previous day’s model for detection.
The model from the previous day is used, so that gained knowledge from
the past is preserved while also not requiring the clients to keep data for


longer periods of time. In general, when training the model from scratch,
more rounds of training are necessary than when adjusting an already existing
model to newer data.

## Metrics
After each round, the clients evaluate the received global model on their test
dataset by computing a set of metrics. Each of these values is aggregated
on the server using a weighted average, where the weights are relative ratios
of the sizes of clients’ data used for generating the metrics. Meaning, that
in the case of testing metrics, the sizes of the test datasets were used, with
the vaccine samples included. The motivation for this is to produce similar
metrics as if they were computed on a complete dataset from all clients.
To evaluate our methods, the metrics used are Accuracy (Acc), True
Positive Rate (TPR), False Positive Rate (FPR)
and F-score. Accuracy is a standard metric used to
evaluate classification and detection models. However, it can not capture
all the relevant information on its own. TPR indicates what ratio of the
malicious samples was detected, and FPR shows how much of the benign
samples are misclassified as malicious. The F-score is often advocated as a
summarizing metric when comparing the performance of two classifiers. We use its unweighted variant F1.

## Comparison to Other Settings
All experiments are repeated ten times with a different random seed for
initializing the model and splitting the dataset. Each run of the federated
experiment is accompanied by training the same model with equivalent
parameters in a local and centralized setting.
. Local setting: The local setting mimics the scenario when the client
decides not to participate in the federated learning and instead creates a
model using only its data. Comparing this to the federated results should
show the benefits of joining the federated process. When evaluating the
local setting, we use the datasets of all clients for the following day. This
is to demonstrate how well the locally trained models would perform
in other clients or when encountering unknown threats. The reported
results are averages of the performance of the individual models.
. Centralized setting: The centralized setting represents a case where
there would be no restrictions on the privacy of the data so that we
could collect all datasets of the clients into a single one and use that
for training the models. This should provide an ideal scenario for the
model’s performance.


# Benefits of Using Our Federated Learning Technique

Our federated learning (FL) approach offers significant advantages in enhancing privacy, scalability, and collaboration while maintaining robust model performance. By enabling distributed model training without sharing raw data, our method ensures data privacy and security, addressing a critical concern in modern data-driven industries. This capability is particularly valuable in sensitive domains such as healthcare, finance, and cybersecurity, where regulatory compliance and ethical considerations demand stringent data protection.

One of the primary benefits of our approach is the ability to leverage the collective knowledge of distributed clients. Each client trains locally on its unique dataset and contributes to a shared global model, which results in a more generalized and diverse representation of the data. This collaborative effort improves model performance, particularly in scenarios where individual datasets are limited or skewed. For instance, in our application of detecting malicious activity in encrypted TLS network traffic, the federated approach enables edge devices to contribute to a global model that performs better than any locally trained model.

Our technique also fosters inclusivity by allowing participation from devices and organizations with varying computational resources. Through optimization methods like FedAvg and FedProx, and by employing regularization to mitigate client drift, our system accommodates a heterogeneous set of participants while ensuring efficient communication and training. This adaptability makes it feasible for both resource-constrained devices and robust organizational servers to participate effectively.

For organizations and individuals considering adoption, our FL approach offers a competitive edge by preserving the value of proprietary datasets. In traditional centralized training, data owners often have to forgo control over their data, risking privacy breaches or intellectual property concerns. With our federated system, participants retain complete ownership of their data, sharing only model updates, which are further protected through regularization and aggregation techniques.

Moreover, our inclusion of mechanisms like "malware vaccines" for clients lacking diverse datasets demonstrates how FL can overcome data heterogeneity challenges. By introducing curated feature vectors without compromising privacy, the system ensures that all participants contribute meaningfully, thus enhancing the overall model quality.

Finally, our FL implementation is designed with practical deployment in mind, utilizing the Flower framework for scalability and flexibility. This framework ensures smooth communication between the central aggregator and clients, allowing for seamless integration into diverse environments. By participating in our FL ecosystem, users gain access to cutting-edge technology that not only enhances their predictive models but also fosters a spirit of cooperation and innovation across industries. Joining this initiative means contributing to a collective intelligence network that balances privacy, security, and performance—essential pillars in today's data-centric landscape.






