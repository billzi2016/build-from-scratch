# build-from-scratch

This is a teaching-oriented repository focused on "building from scratch," used to systematically organize scratch implementations of algorithms, models, tools, or workflows.

The goal of this project is not to simply call existing libraries and produce results, but to present the core processes, key data structures, and execution steps as clearly as possible so readers can genuinely understand the implementation path.

## Project Goals

This repository is mainly intended to solve four things:

- Complete common topics with as few external dependencies as possible
- Distinguish between "demo-level usage" and "underlying scratch implementation"
- Keep the code structure clear, the process runnable, and the project suitable for learning and demonstration
- Help beginners understand not only how to use something, but also how it is built underneath

## Design Principles

The repository follows these principles overall:

- Teaching first, without showing off with flashy code
- Emphasize observable process rather than only giving the final answer
- Implement core logic ourselves as much as possible, instead of outsourcing key steps to third-party libraries
- Think through the architecture before coding, so the codebase does not turn into a pile of patches

## Current Directories

- `root_prd.md`: repository-level overall specification, defining global goals, directory conventions, implementation layers, and dependency boundaries
- `bpe/`: scratch implementations and documentation related to BPE
- `imagenet-ai/`: implementation and experiment materials around the ImageNet AI topic

## Repository Structure Convention

For each specific topic, the project is generally split into two layers:

1. Outer demo layer  
   Used to show how to run and how to use it
2. Scratch implementation layer  
   The actual underlying implementation, responsible for building the core logic step by step

The purpose of this design is to avoid wrapping a mature library with a thin shell and then incorrectly calling it "from scratch."

## Who This Is For

This repository is suitable for:

- People who want to understand algorithms or models from the ground up
- People who want to create teaching-oriented Notebooks, demos, or small projects
- People who are not satisfied with just "knowing how to call it" and want to understand the implementation process

## Suggested Reading Order

It is recommended to read in the following order:

1. Start with `root_prd.md` to understand the overall design of the repository
2. Then enter a specific topic directory and read its corresponding `README.md` and `*_prd.md`
3. Finally, run the implementation itself and observe the intermediate process and outputs

## Repository Positioning Summary

`build-from-scratch` is more like a long-term maintained "build-from-scratch laboratory":

- It emphasizes teaching clarity
- It emphasizes structured implementation
- It emphasizes breaking down complex things and explaining them clearly

Rather than only pursuing "it runs" or "the result is correct."
