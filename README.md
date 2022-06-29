**Language models** will **transform** the **design** and
**implementation** of **programming languages**. This repository is a
collection of **experiments** in combining language models and
programming languages.

![Three synthesized images of parrots.](./bin/parrot.png)

- [Introduction](#introduction)
- [Languages](#languages)
  - [Algebraic Data Types](#algebraic-data-types)
    - [Reversible Computing](#reversible-computing)
  - [Deep Inference](#deep-inference)
    - [Subatomic Logic](#subatomic-logic)
  - [Computability Logic](#computability-logic)
  - [Open Games](#open-games)
    - [Generative Adversarial Networks](#generative-adversarial-networks)
  - [Signal Flow Diagrams](#signal-flow-diagrams)
    - [Geometric Algebra](#geometric-algebra)
  - [Constructive Solid Geometry](#constructive-solid-geometry)
  - [Functional Reactive Programming](#functional-reactive-programming)
  - [Lambda Calculus](#lambda-calculus)
    - [The Lambda Cube](#the-lambda-cube)
- [Synthesizers](#synthesizers)
  - [Continuation, Diffusion, and Energy Models](#continuation-diffusion-and-energy-models)
  - [Transformers](#transformers)
  - [State Space Models](#state-space-models)
- [Experiments](#experiments)
- [Related Work](#related-work)
- [References](#references)
  - [Evolution through Large Models](#evolution-through-large-models)
  - [Search and Representation in Program Synthesis](#search-and-representation-in-program-synthesis)
  - [Decision Transformer](#decision-transformer)
  - [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](#mastering-atari-go-chess-and-shogi-by-planning-with-a-learned-model)
  - [Algebraic Information Effects](#algebraic-information-effects)
  - [An Algorithmic Interpretation of a Deep Inference System](#an-algorithmic-interpretation-of-a-deep-inference-system)

# Introduction


I'd like to make an **analogy** between **synthesis** and standard
language services like **garbage collection**, **event loops**,
**package management**, and so on.

# Languages
In general I'm interested in using **small**, **typed** languages for
synthesis experiments. In particular I'm interested in **deep
inference**, **computability logic**, and **compact closed
categories**.

## Algebraic Data Types
**Product and sum types** are a standard part of functional
languages. You can **derive** a functional language from their
properties, like associativity, commutativity, and the distributive
law:

```
     id:            A <-> A                  :id

 unite*:        1 * A <-> A                  :uniti*
  swap*:        A * B <-> B * A              :swap*
assocl*:  A * (B * C) <-> (A * B) * C        :assocr*

 unite+:        0 + A <-> A                  :uniti+
  swap+:        A + B <-> B + A              :swap+
assocl+:  A + (B + C) <-> (A + B) + C        :assocr+

absorbr:        0 * A <-> 0                  :factorzl
   dist:  (A + B) * C <-> (A * C) + (B * C)  :factor
```

These are the **primitive combinators**. To compose these, you can use
**products and sums**, as well as **sequencing**:

```
    f:     A <-> B
    g:     C <-> D
---------------------- product
f * g: A * C <-> B * D
   
    f:     A <-> B
    g:     C <-> D
---------------------- sum
f + g: A + C <-> B + D

    f:     A <-> B 
    g:     B <-> C
---------------------- sequence
f ; g:     A <-> C
```
### Reversible Computing

## Deep Inference
The **Curry-Howard-Lambek correspondence** allows you to **derive** a
**programming language** from any **logical system**. [**Deep
inference**](http://alessio.guglielmi.name/res/cos/) makes this
straightforward, by **composing** **programs** and **types** with the
**same operators**.

### Subatomic Logic
**Subatomic logic** allows you to **generate a type system** (and
therefore a programming language) from a logical schema.

## Computability Logic

## Open Games

### Generative Adversarial Networks

## Signal Flow Diagrams

### Geometric Algebra

## Constructive Solid Geometry

## Functional Reactive Programming

## Lambda Calculus

### The Lambda Cube
It might be interesting to try some sort of geometric analysis of [the
lambda cube](https://en.wikipedia.org/wiki/Lambda_cube).

# Synthesizers

## Continuation, Diffusion, and Energy Models

## Transformers

## State Space Models

# Experiments

# Related Work

# References

## [Evolution through Large Models](https://arxiv.org/abs/2206.08896)

> This paper pursues the insight that large language models (LLMs)
> trained to generate code can vastly improve the effectiveness of
> mutation operators applied to programs in genetic programming
> (GP). Because such LLMs benefit from training data that includes
> sequential changes and modifications, they can approximate likely
> changes that humans would make. To highlight the breadth of
> implications of such evolution through large models (ELM), in the
> main experiment ELM combined with MAP-Elites generates hundreds of
> thousands of functional examples of Python programs that output
> working ambulating robots in the Sodarace domain, which the original
> LLM had never seen in pre-training. These examples then help to
> bootstrap training a new conditional language model that can output
> the right walker for a particular terrain. The ability to bootstrap
> new models that can output appropriate artifacts for a given context
> in a domain where zero training data was previously available
> carries implications for open-endedness, deep learning, and
> reinforcement learning. These implications are explored here in
> depth in the hope of inspiring new directions of research now opened
> up by ELM.

## [Search and Representation in Program Synthesis](https://dspace.mit.edu/handle/1721.1/143375)

> Building systems that can synthesize programs from natural
> specifications (such as examples or language) is a longstanding goal
> of AI. Building such systems would allow us to achieve both
> scientific and practical goals. From a scientific perspective,
> program synthesis may provide a way to learn compact, generalizable
> rules from a small number of examples, something machine learning
> still struggles with, but humans find easy. From a practical
> perspective, program synthesis systems can assist with real-world
> programming tasks, from novice end-user tasks (such as string
> editing or repetitive task automation) to expert functions such as
> software engineering. In this work, we explore how to build such
> systems. We focus on two main interrelated questions: 1) When
> solving synthesis problems, how can we effectively search in the
> space of programs and partially constructed programs? 2) When
> solving synthesis problems, how can we effectively represent
> programs and partially constructed programs? In the following
> chapters, we will explore these questions. Our work has centered
> around the syntax and the semantics of programs, and how syntax and
> semantics can be used as tools to assist both the search and
> representation of programs and partial programs. We present several
> algorithms for synthesizing programs from examples, and demonstrate
> the benefits of these algorithms over previous approaches.

## [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

> We introduce a framework that abstracts Reinforcement Learning (RL)
> as a sequence modeling problem. This allows us to draw upon the
> simplicity and scalability of the Transformer architecture, and
> associated advances in language modeling such as GPT-x and BERT. In
> particular, we present Decision Transformer, an architecture that
> casts the problem of RL as conditional sequence modeling. Unlike
> prior approaches to RL that fit value functions or compute policy
> gradients, Decision Transformer simply outputs the optimal actions
> by leveraging a causally masked Transformer. By conditioning an
> autoregressive model on the desired return (reward), past states,
> and actions, our Decision Transformer model can generate future
> actions that achieve the desired return. Despite its simplicity,
> Decision Transformer matches or exceeds the performance of
> state-of-the-art model-free offline RL baselines on Atari, OpenAI
> Gym, and Key-to-Door tasks.

## [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)

> Constructing agents with planning capabilities has long been one of
> the main challenges in the pursuit of artificial
> intelligence. Tree-based planning methods have enjoyed huge success
> in challenging domains, such as chess and Go, where a perfect
> simulator is available. However, in real-world problems the dynamics
> governing the environment are often complex and unknown. In this
> work we present the MuZero algorithm which, by combining a
> tree-based search with a learned model, achieves superhuman
> performance in a range of challenging and visually complex domains,
> without any knowledge of their underlying dynamics. MuZero learns a
> model that, when applied iteratively, predicts the quantities most
> directly relevant to planning: the reward, the action-selection
> policy, and the value function. When evaluated on 57 different Atari
> games - the canonical video game environment for testing AI
> techniques, in which model-based planning approaches have
> historically struggled - our new algorithm achieved a new state of
> the art. When evaluated on Go, chess and shogi, without any
> knowledge of the game rules, MuZero matched the superhuman
> performance of the AlphaZero algorithm that was supplied with the
> game rules.

## [Algebraic Information Effects](https://scholarworks.iu.edu/dspace/handle/2022/26738)

> From the informational perspective, programs that are usually
> considered as pure have effects, for example, the simply typed
> lambda calculus is considered as a pure language. However,
> β–reduction does not preserve information and embodies information
> effects. To capture the idea about pure programs in the
> informational sense, a new model of computation — reversible
> computation was proposed. This work focuses on type-theoretic
> approaches for reversible effect handling. The main idea of this
> work is inspired by compact closed categories. Compact closed
> categories are categories equipped with a dual object for every
> object. They are well-established as models of linear logic,
> concurrency, and quantum computing. This work gives computational
> interpretations of compact closed categories for conventional
> product and sum types, where a negative type represents a
> computational effect that “reverses execution flow” and a fractional
> type represents a computational effect that “allocates/deallocates
> space”.

## [An Algorithmic Interpretation of a Deep Inference System](https://link.springer.com/chapter/10.1007/978-3-540-89439-1_34)

> We set out to find something that corresponds to deep inference in the
> same way that the lambda-calculus corresponds to natural
> deduction. Starting from natural deduction for the
> conjunction-implication fragment of intuitionistic logic we design a
> corresponding deep inference system together with reduction rules on
> proofs that allow a fine-grained simulation of beta-reduction.
