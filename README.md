# Stochastic Parrots

**Stochastic Parrots**, also known as **Napkin Norns**, is a small experiment in
making text-processing programs behave more like persistent software organisms
than like chat completions.

The word *norn* is intentionally playful. A norn in this repository has needs,
mood, memory, a changing environment, and a filesystem presence. Those are
not claims of biological consciousness; they are names for explicit state and
rules that can be inspected in the source.

The project is also a practical distinction between two ideas that are often
collapsed into one:

> A system can produce stochastic, language-shaped output without being a
> language model or a probabilistic next-token prediction engine.

## The short version

A norn:

- consumes text into a finite, persistent collection of fragments;
- builds word-frequency and word-pair tables;
- recombines, splices, shuffles, and mutates those fragments with Python's
  random number generator;
- maintains a symbolic grid that text can perturb;
- changes needs, mood, energy, curiosity, and personality entropy over time;
- saves its memory, grid, and counters to JSON;
- can run as an interactive process or a background daemon communicating
  through files; and
- can optionally use an embedding service to choose memories that are
  semantically near a topic.

It does **not** contain a neural network, model weights, a tokenizer, a
next-token distribution, an attention stack, or a training loop. Its text is
assembled from state it already has and from explicitly coded transformations.

## What is a norn?

`NapNorn` is the organism-shaped layer in [`Napnorn.py`](Napnorn.py). It
coordinates three kinds of state:

1. **Experience and memory** — every perception is kept as a memory fragment
   (up to 1,000 fragments), and is also consumed by the `MLBabel` text engine.
2. **An environment** — `MLWastesSwarm` maintains a 200×40 symbolic map. Input
   text is converted into symbols and an intensity, then used to perturb
   regions of the map.
3. **Internal variables** — hunger, energy, social need, curiosity,
   consciousness level, personality entropy, mood, and thought count.

The organism's behavior is a loop of state transitions rather than a single
stateless request/response call:

```text
perceive input
    ├─ consume input into fragment memory and word associations
    ├─ perturb the symbolic grid
    ├─ append the experience to persistent memory
    └─ slightly increase consciousness_level

update_needs
    ├─ decay needs as wall-clock time passes
    └─ derive mood from needs and consciousness

think
    ├─ recall memories through the grid, or semantic focus when enabled
    ├─ ask MLBabel to dream a text fragment
    └─ feed the thought back into the grid
```

The vocabulary is deliberately theatrical: “hunger” means a numeric drive for
new input, “sleep” means a command or shutdown path that preserves state, and
“consciousness” is a counter-like value adjusted by the program. The important
point is that these properties are explicit, local, and inspectable.

## The mechanical difference from a language model

“Language model” can mean several things in everyday conversation, so this
comparison is about the usual neural autoregressive model: a learned function
that maps a token context to a probability distribution for the next token.

| Dimension | A typical language model | A norn in this repository |
| --- | --- | --- |
| **Basic operation** | Computes scores/probabilities for possible next tokens. | Selects stored fragments or words, then applies coded recombination operations. |
| **Where knowledge lives** | Distributed numerical parameters (weights). | JSON memory, word frequencies, word-pair lists, grid symbols, and scalar state. |
| **Learning** | Usually gradient-based optimization over a training corpus, often before deployment. | Runtime ingestion: append fragments, count words, record pairs, perturb a grid, and adjust state variables. No gradients. |
| **Generation unit** | Usually one token at a time, conditioned on the generated prefix. | A line assembled by fragment splicing, Markov-like pair following, word salad, shuffling, or grid-driven recall. |
| **Probability mechanism** | A model-derived distribution such as softmax logits. | Calls to `random.choice`, `random.sample`, `random.shuffle`, `random.randint`, and related operations. |
| **Memory** | Context window plus learned weights; persistent memory requires an external system. | Explicit bounded lists that can be saved, loaded, inspected, trimmed, and bred. |
| **Identity over time** | Normally supplied by the surrounding application and conversation state. | A named object with needs, mood, age, counters, files, and a saved brain. |
| **Environment** | Usually receives an input sequence and returns output. | Text changes a persistent symbolic map; the map influences later thought. |
| **Failure mode** | Can produce fluent but unsupported continuations from its learned distribution. | Can run out of memory, have an empty grid, fall back to random selection, or preserve a literal bug in its hand-written rules. |
| **Determinism** | Depends on model, sampler, seed, runtime, and implementation. | A seed can make the Python random choices reproducible for a given state and execution path. |

The norn can be stochastic without predicting the probability of language in
the modeling sense. Its randomness answers questions such as “which stored
fragment should I choose?” and “where should these words be cut?” It is not
estimating a probability distribution over all possible next tokens.

### `MLBabel` is a recombination engine

[`MLBabel.py`](MLBabel.py) is the norn's text machinery. It has three main
inputs: its current fragment memory, its association tables, and an entropy
number that chooses a generation regime.

- **Low entropy** mostly returns an existing fragment, occasionally with a
  small word swap or a merge of two fragments.
- **Medium entropy** follows recorded adjacent-word pairs or splices slices
  from several fragments.
- **High entropy** draws words from the frequency table, deeply shuffles a
  fragment, or folds slices from multiple fragments into a word soup.

The pair-following routine is described in the code as “Markov-like”, but that
does not turn the program into a trained language model. It is a small table of
observed adjacent words and a random walk through that table. There are no
learned parameters, token logits, hidden states, or probability estimates
conditioned by a neural network.

### Embeddings are optional attention, not generation

If `NORN_EMBED_URL` is configured, `embed_client.py` can call an
OpenAI-compatible embedding endpoint. `MLBabel` stores the returned vectors and
uses cosine similarity plus a temperature derived from entropy to bias which
existing fragment is selected.

That optional service changes **recall**, not the nature of generation:

- it does not generate text;
- it does not replace the norn's memory or rules;
- it does not add model weights to this repository; and
- if the endpoint fails, the client returns `None` and selection falls back to
  ordinary random choice.

This makes semantic focus a useful adapter around a norn, not evidence that the
norn itself is a probabilistic prediction engine.

## Norn anatomy

### Memory and perception

`NapNorn.perceive()` sends input through all three state channels:

1. `MLBabel.consume()` strips and splits the text, truncates fragments to a
   target size, stores them, and updates word statistics.
2. `MLWastesSwarm.perturb_map()` detects keywords and punctuation, chooses
   symbols and zones, and may trigger a larger map event after enough input.
3. The original input is appended to `memory_fragments`, capped at 1,000
   entries.

The same experience therefore has both a textual representation and a spatial
representation. The norn does not “understand” the input through an opaque
latent vector by default; it records it and applies visible rules.

### Thought

Without an embedder, `think()` samples twelve random locations in the grid,
turns active symbols into keyword sets, finds memories containing those
keywords, and gives up to seven matching memories to a temporary `MLBabel`.
If there are no matches, it samples up to five memories instead.

With an embedder, the norn combines recent perceptions with keywords associated
with active grid symbols, embeds that focus text, and lets its main `MLBabel`
bias fragment recall toward the focus. Either way, the final text still comes
from the Babel transformations described above.

After every thought, the thought itself perturbs the grid. This feedback loop
is the project's “recursive consciousness”: a literal state transition in
which output becomes a future environmental influence.

### Needs and mood

Needs begin at 100 and decay on a wall-clock schedule, subject to the minimums
encoded in `update_needs()`. Mood is calculated from average needs and the
consciousness level. Depending on those values, a norn can be happy,
questioning, philosophical, distressed, and so on.

The daemon can act without a human command:

- hunger triggers self-feeding from a memory or a meta-thought;
- low energy can trigger rest;
- curiosity can trigger automatic thinking; and
- urgent needs produce focused expressions from related memories.

These behaviors are not emergent optimization. They are deterministic
conditionals containing random choices, which makes them straightforward to
inspect and modify.

### Persistence and file presence

The daemon stores a brain and publishes status in `norn_brains/` by default:

```text
norn_brains/
├── Sparkle_brain.json       # memory, needs, grid, counters, associations
├── Sparkle_grid.json        # MLWastesSwarm's map state
├── Sparkle_status.json      # current status for external readers
├── Sparkle_command.txt     # one-shot command input
└── Sparkle_response.txt    # response to the last command
```

Brain files preserve the norn's explicit state. Embedding vectors are
intentionally not persisted; when a saved brain is loaded with an embedder,
the fragments are re-embedded. A brain can therefore be inspected with normal
JSON tools rather than requiring a model runtime.

## Running it

The implementation uses the Python standard library. From the repository root:

```bash
python Napnorn.py interactive
```

The interactive commands are:

```text
feed:<text>   Give the norn an experience
pet           Show affection
play          Engage in play
think         Make the norn think
sleep         Help the norn rest
status        Get a brief status summary
report        Get the detailed state report
save          Save the brain
quit          Save and exit
```

Run a file-backed daemon instead:

```bash
python Napnorn.py daemon Sparkle
```

Then send a command from another terminal:

```bash
echo 'feed:The machine remembers the shape of rain' \
  > norn_brains/Sparkle_command.txt
cat norn_brains/Sparkle_status.json
cat norn_brains/Sparkle_response.txt
```

The daemon checks the command file, updates needs, performs possible
auto-actions, writes status, and periodically saves the brain. Stop it with
`Ctrl+C` to save once more before it goes to sleep.

### Using the Babel engine directly

`MLBabel.py` can also be used as a command-line fragment engine:

```bash
python MLBabel.py notes.txt --lines 10 --entropy 0.5
python MLBabel.py notes.txt --entropy 0.8 --seed 42
cat notes.txt | python MLBabel.py --oracle "What is truth?"
cat notes.txt | python MLBabel.py --stream
```

The `--seed` option is useful when examining how a particular state produces
its output. It makes the experiment repeatable without changing the underlying
mechanism.

## Optional semantic focus

Semantic focus is opt-in. The default behavior has no embedding dependency and
uses the original random/grid-driven recall path.

For a local Ollama-compatible server:

```bash
ollama pull all-minilm
export NORN_EMBED_URL=http://127.0.0.1:11434/v1/embeddings
export NORN_EMBED_MODEL=all-minilm:latest
python Napnorn.py interactive
```

`embed_client.py` uses only `urllib`, sends one embedding request at a time,
and fails open on timeouts, bad responses, or unavailable servers. The norn
continues thinking when semantic focus is unavailable.

## Breeding norns

Two saved brains can produce a child:

```bash
python Napnorn.py breed \
  norn_brains/A_brain.json \
  norn_brains/B_brain.json \
  Child
```

Breeding samples memories from both parents, optionally drops or duplicates a
word in a fragment, averages personality entropy with a little jitter, and
feeds the inherited memories into a fresh norn. With `--grid-osmosis`, the
parents' symbolic maps are resampled and blended into the child's birth
terrain:

```bash
python Napnorn.py breed A_brain.json B_brain.json Sprout \
  --memories 60 --mutation 0.05 --grid-osmosis
```

This is “genetics” as explicit list sampling and mutation, not biological
reproduction and not parameter fine-tuning.

## What this project is—and is not

**It is:**

- a small, inspectable experiment in persistent state;
- a text recombination and symbolic-environment toy;
- a way to explore how memory, feedback, timers, and interfaces can create
  organism-like behavior;
- stochastic in the ordinary programming sense; and
- intentionally poetic about its implementation.

**It is not:**

- a neural language model;
- a replacement for an LLM or a general-purpose conversational assistant;
- evidence that random text is intrinsically meaningful;
- a claim that a JSON file is a biological brain; or
- proof that any particular software system is conscious.

The useful distinction is mechanical, not rhetorical. Calling the output a
“stochastic parrot” can describe its style, but it does not tell us how it was
made. In this repository, the parrot is not a hidden probability engine. It is
a stateful collection of lists, counters, symbols, timers, and random
operations that happens to dream in text.

## Files at a glance

| File | Role |
| --- | --- |
| [`Napnorn.py`](Napnorn.py) | Norn lifecycle, needs, mood, perception, thinking, persistence, daemon, and breeding. |
| [`MLBabel.py`](MLBabel.py) | Fragment memory, word associations, entropy-controlled recombination, streaming, and oracle mode. |
| [`MLWastes.py`](MLWastes.py) | Persistent symbolic grid, text pattern matching, perturbations, and map statistics. |
| [`embed_client.py`](embed_client.py) | Optional fail-open client for embedding-based memory selection. |
| [`LICENSE`](LICENSE) | Unlicense/public-domain dedication. |

## License

This project is released under the [Unlicense](LICENSE).
