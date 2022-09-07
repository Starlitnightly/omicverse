# Developer guild

> **Note**
>
> To better understand the following guide, you may check out our [publication]() first to learn about the general idea.

Below we describe main components of the framework, and how to extend the existing implementations.

## Main components

A Pyomic framework is primarily composed of 2 components.

- bulk omic: to analysis the bulk omic-seq like RNA-seq or Proper-seq.
- single cell omic: to analysis the single cell omic-seq like scRNA-seq or scATAC-seq

New extensions can be added to `bulk` or `single` as well.