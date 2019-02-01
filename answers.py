r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
    batch_size=100,
    h_dim=128, z_dim=64, x_sigma2=0.1,#0.5-good #0.9,
    learn_rate=0.001, betas=(0.5,0.5))#(0.9, 0.999))
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=20, z_dim=1000,#100 is ok,
        data_label=1, label_noise=0.15,#0.15 is ok
        discriminator_optimizer=dict(
            type='SGD',
            lr=0.001, #0.005 is ok,
        ),
        generator_optimizer=dict(
            type='Adam',
            lr=0.001,
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
Training GAN is actually two phases:
Training the Discriminator (D), and training the Generator (G).
    -   To do one step in optimizer of D we need to provide fake and real data.
        Real data is from dataset, and the fake data which is sampled from G.
        In this stage we optimize only D, so when we sample fake data from G,
        we don't need the maintain the gradient of the noise which pass through G.
        We just need the generated fake image.
        
    -   On the other hand, when training G, we obviously need to maintain the
        gradient, while the noise passes through the generator. Without doing
        it, we could not train the generator.  

"""

part3_q2 = r"""

1.  No. In general, training GAN is finding an equilibrium of two competing losses.
    So even if Gen loss is below some threshold, it still can produce visually "bad" images.
    For example, we have a Disc. which is not training well, or too slowly. Also, we
    have a Gen. which training fast, and produces better images which can "fool" the Disc.
    each epoch. In some point our Gen. will produce perfect fakes to fool our "stupid" 
    Disc. and its loss will drop below some threshold. But as our Disc. is malfunctioning, 
    the Gen. didn't actually learned to produce fake images which looks like real world
    data.
    
2.  If Gen. loss decrease, that means that it could produce better images which can fool 
    the Disc. 
    On the other hand, Disc. loss is constant. As we know Disc. loss is a sum of two losses,
    one for real data, and one for fake data.
    In this case, loss of fake data increased (as Disc. fooled more time),
    But loss of real data decreased, as a result of Disc. getiing better in classifying real
    world data. The sum of the losses remains constant.   

"""

part3_q3 = r"""
The main difference between two results is:
-   VAE output is unrealistic and blurry, 
    while well-trained GAN output is sharper and more realistic. 

The reason is VAE learn an complex distribution of the data 
by trying to fit the data via a Gaussian distribution.
On the other hand, well-trained GAN generates images
from the same distribution of the real world data. 
"""

# ==============


