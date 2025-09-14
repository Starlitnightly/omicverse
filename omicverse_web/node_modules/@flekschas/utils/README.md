# A collection of handy utility functions

[![NPM Version](https://img.shields.io/npm/v/@flekschas/utils.svg?style=flat-square&color=7f99ff)](https://npmjs.org/package/@flekschas/utils)
[![Build Status](https://img.shields.io/github/actions/workflow/status/flekschas/utils/build.yml?branch=master&color=a17fff&style=flat-square)](https://github.com/flekschas/utils/actions?query=workflow%3Abuild)
[![File Size](http://img.badgesize.io/https://unpkg.com/@flekschas/utils/dist/utils.min.js?compression=gzip&style=flat-square&color=e17fff)](https://bundlephobia.com/result?p=@flekschas/utils)
[![Code Style Prettier](https://img.shields.io/badge/code%20style-prettier-ff7fe1.svg?style=flat-square)](https://github.com/prettier/prettier#readme)
[![Docs](https://img.shields.io/badge/api-docs-ff7fa5.svg?style=flat-square)](API.md)

This is a collection of utility functions that I keep using across different
projects. I primarily created this package for myself so I don't have to
re-implement certain functions over and over again, and to have a central place for testing them.

## Install

```bash
npm install @flekschas/utils --save-dev
```

## Usage

```javascript
import { debounce } from '@flekschas/utils';

const hi = debounce(() => {
  console.log('I am debounced');
}, 250);
```

For cherry picking from a specific topic do:

```javascript
import { debounce } from '@flekschas/utils/timing';
```

The utility functions are organized by the following topics:

- animation
- color
- conversion
- dom
- event
- functional-programming
- geometry
- map
- math
- object
- other
- sorting
- string
- timing
- type-checking
- vector

## API

See [API.md](API.md) for the API docs.

## Why yet another library for utility functions?

Generally, I follow four core goals with this collection:

1. Reusability
2. Performance
3. Simplicity
4. No dependencies

Whenever a function is _reusable in a general context_ I might add it. When I
add a function I will make sure it's _performant_. Finally, every function
should be implement as _simple as possible_ without harming performance.
There's always a trade-off between performance and simplicity and my philosophy
is the following: if the simple and complex implementation perform roughly the
same, I choose the simple implementation. If a slightly more complex
implementation is much faster I will favor the complex implementation. In any
case, the API should always be simple and easy to understand! Finally,
I want my utils functions to have no external and as little as possible
internal dependencies. Why? No matter how large this collection becomes as a whole,
if you only need one function, you should only ever have to bundle
a single function and not a whole forrest of depending helper functions.
