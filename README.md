experiments with programming languages and language models

---

i want to use a couple of typed, functional languages for
synthesis. im interested in deep inference, computability logic,
compact closed categories.

---

- [Algebraic Information Effects](https://scholarworks.iu.edu/dspace/handle/2022/26738)

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

- [Search and Representation in Program Synthesis](https://dspace.mit.edu/handle/1721.1/143375)

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

- [Evolution through Large Models](https://arxiv.org/abs/2206.08896)

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

- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

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

- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)

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
