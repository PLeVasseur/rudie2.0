# Rudie2.0

An experimental re-imagining of the [rudie](https://github.com/PLeVasseur/rudie)
library for Kalman Filters, still `#![no_std]`.

## Status

Pre-pre-pre-alpha

* The architecture is in place
* Would like to flesh out library of measurement models and
  process models

 ## Philosophy
* Make use of Rust's const generics (previous version in 2018 before they
  were available)
* Available for embedded devices without an RTOS
* No dynamic heap memory allocations, pre-allocated within Workspaces
* Attempt to make a "toolkit", where it's possible to have shared building
  blocks throughout the library and you can reach down the level of
  specificity needed for your problem
* By having loose coupling between measurement models / process models and
  the particular Kalman filter state space and overall measurement space
  this would allow us to have a library of different measurement and
  process models that could be used in different configurations.

## Library Architecture

![rudie2.0_architecture.png](diagrams/rudie2.0_architecture.png)