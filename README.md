# Rudie2.0

An experimental re-imagining of the [rudie](https://github.com/PLeVasseur/rudie)
library for Kalman Filters.

 ## Philosophy
* Make use of Rust's const generics (previous version in 2018 before they
  were available)
* Attempt to make a "toolkit", where it's possible to have shared building
  blocks throughout the library and you can reach down the level of
  specificity needed for your problem