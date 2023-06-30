# Developer guild

> **Note**
>
> To better understand the following guide, you may check out our [publication](https://doi.org/10.1101/2023.06.06.543913) first to learn about the general idea.

Below we describe main components of the framework, and how to extend the existing implementations.

## Main components

A omicverse framework is primarily composed of 3 components.

- bulk omic: to analysis the bulk omic-seq like RNA-seq or Proper-seq.
- single cell omic: to analysis the single cell omic-seq like scRNA-seq or scATAC-seq

New extensions can be added to `bulk`, `single` or `bulk2single` as well.