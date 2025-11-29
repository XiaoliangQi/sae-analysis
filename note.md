# Note on the analysis of sparse auto-encoder

In this note I want to describe the idea that we are pursuing in this project.

## Main idea

We want to compare features in SAE to bulk operators in AdS/CFT. They have the following common properties:
- They are nonlocal in term of input tokens (which are the analog of operators on the boundary in AdS/CFT).
- They are supposed to be mostly independent (which corresponds to emergent locality in AdS/CFT). This is only true for the given probability distribution of input text. In the abstract sense (if we input arbitrary token vectors) they are not independent.
- They are sparse (which is the analog of the fact that in AdS/CFT semi-classical spacetime is only meaningful if the number of bulk excitation is small.)

Then motivated by this analogy, we would like to ask the following questions:
1. What is the analog of operator size? Is there a hierachy of "size" for the features (some features are more nonlocal than others). This seems to be related to the discussion in Ref.[1].
2. What is the analog of bulk geometry? Can we compute the dynamics or correlation of features to obtain the geometry of the bulk?
3. Can we use AdS/CFT ideas such as HKLL to define a new type of feature extraction? In particular, thinking about the dynamics (in time and in token space), is there a better way to define features?
4. Information measure of SAE. Similar to the AdS/CFT case is there some measure such as entropy, relative entropy, which have geometric meaning? Is there a measure that tells us the SAE is a good "compression"?

## More details

### Operator Size Distribution

We can define the number of tokens that lead to activation of a given feature. Denote the input token vectors as $x_i$, with $i$ labeling all tokens, and they are input in batches $B_I$ (which are paragraphs). $B_I={x_{I_1}x_{I_2}...x_{I_{L_I}}}$. Then the transformer maps $B_I$ to an intermediate layer 

$\left(y_1,y_2,...,y_{L_I}\right)=T(B_I)$

The spare-auto-encoder is defined by a nonlinear map 

$f_i=E(y_i)$ with $f_i$ a high-dimensional sparse vector.

The decoder is a linear map

$\tilde{y}_i=D(f_i)$

The number of tokens that activates a feature is

$N_a=\left|i: \sum_{I,~\text{s.t.}~x_i\in B_I}\theta(|f_i^a|-f_{\rm min})>0\right|$
which counts how many unique tokens leads to activation of feature $a$ on the same location of this token in some of the text batches $B_I$. Then we can study the statistics of $N_a$ in the space of features. This is analog of operator size in the sense of how many "local operators" are strongly correlated with feature $a$. 

### Correlation between features

To understand the "geometry" of feature space, we can compute correlation function of the features in the given dataset. Define $A_i^a(B_I)\theta(|f_i^a|-f_{\rm min})$ as a binary variable of whether feature $a$ is activated in $B_I$ on token $i$, we can define

$
\left\langle A^a\right\rangle=\frac1{N_B}\sum_{B_I}\frac1{L_I}\sum_i A_i^a(B_I)
$

and

$
\left\langle A^aA^b\right\rangle=\frac1{N_B}\sum_{B_I}\frac1{L_I}\sum_i A_i^a(B_I)A_i^b(B_I)
$

Then the connection correlation is defined as

$
C_{ab}=\left\langle A^aA^b\right\rangle-\left\langle A^a\right\rangle\left\langle A^b\right\rangle$


## References

1. Bussmann, B., et al. (2025). “Learning Multi-Level Features with Matryoshka Sparse Autoencoders.” (ICML 2025), arXiv 2503.17547 (https://arxiv.org/pdf/2503.17547)
2. Chanin, D., et al. (2024). “A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders.” (arXiv 2409.14507 https://arxiv.org/pdf/2409.14507)
3. Chanin, D., Dulka, T., & Garriga-Alonso, A. (2025). “Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders.” (arXiv 2505.11756 https://arxiv.org/pdf/2505.11756)
4. Bricken, T., et al. (2023). “Towards Monosemanticity: Decomposing Language Models with Dictionary Learning.” (Transformer Circuits Thread) https://transformer-circuits.pub/2023/monosemantic-features/
5. (Scaling in correlation matrix distribution) https://arxiv.org/pdf/2410.19750
6. (Zipf law) https://arxiv.org/pdf/2411.02124 
7. (Scaling law of error vs model size) https://arxiv.org/abs/2509.02565
8. (Introduces a routing mechanism to share an SAE across multiple transformer layers; observes that feature activation patterns vary by layer, with “low-level” features peaking early and “high-level” features later – consistent with a heavy-tailed distribution of feature abstraction across layers) https://arxiv.org/abs/2503.08200