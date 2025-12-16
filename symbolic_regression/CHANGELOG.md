# Changelog

## [0.4.0](https://github.com/MilesCranmer/symbolic_regression.rs/compare/symbolic_regression-v0.3.0...symbolic_regression-v0.4.0) (2025-12-16)


### ⚠ BREAKING CHANGES

* move `compress_constants` to de

### Features

* add simplification utilities ([06f738c](https://github.com/MilesCranmer/symbolic_regression.rs/commit/06f738cece71f0a3923f3f2e520a8d418182ac17))
* additional loss functions ([78e3da2](https://github.com/MilesCranmer/symbolic_regression.rs/commit/78e3da29110fc80f3b5caf72c5fca24ab3f94587))
* batched dataset ([5483597](https://github.com/MilesCranmer/symbolic_regression.rs/commit/5483597ab8f4b5c8191028c638335a1d8499e402))
* custom complexities ([c0f5b31](https://github.com/MilesCranmer/symbolic_regression.rs/commit/c0f5b31d51603d053fe994709122df5bcf3dc5ec))
* move `compress_constants` to de ([2031494](https://github.com/MilesCranmer/symbolic_regression.rs/commit/203149490cb786b9fb95a6b8fb7c73b51bfc1c6a))
* simplify by default ([8dd6332](https://github.com/MilesCranmer/symbolic_regression.rs/commit/8dd6332e4cc36d71006747bb1233852ef4704f9d))
* wasm compatibility ([67c071f](https://github.com/MilesCranmer/symbolic_regression.rs/commit/67c071f5d8cbbdb3760e2293a42019e237a95290))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * dynamic_expressions bumped from 0.3.0 to 0.4.0

## [0.3.0](https://github.com/MilesCranmer/symbolic_regression.rs/compare/symbolic_regression-v0.2.0...symbolic_regression-v0.3.0) (2025-12-15)


### ⚠ BREAKING CHANGES

* upgrade core deps
* add command line interface

### deps

* upgrade core deps ([3b57512](https://github.com/MilesCranmer/symbolic_regression.rs/commit/3b57512637955c6fab0e5fe2c65b8f93455d94b1))


### Features

* add command line interface ([77f4b5e](https://github.com/MilesCranmer/symbolic_regression.rs/commit/77f4b5eda8d072f89915db6b5f0eddcce9138538))
* more ergonomic CLI ([8475acd](https://github.com/MilesCranmer/symbolic_regression.rs/commit/8475acdda6059db82b08a2a20054882b2721f32f))
* more robust BFGS ([258d599](https://github.com/MilesCranmer/symbolic_regression.rs/commit/258d599fb5741205eafe5742598442c536960055))
* use operator registry from sr ([52ff642](https://github.com/MilesCranmer/symbolic_regression.rs/commit/52ff6425ab7ff90b071083e44105533455a90d7d))

## [0.2.0](https://github.com/MilesCranmer/symbolic_regression.rs/compare/symbolic_regression-v0.1.0...symbolic_regression-v0.2.0) (2025-12-14)


### ⚠ BREAKING CHANGES

* DRY principles and better hierarchy

### Features

* create dynamic_expressions.rs and symbolic_regression.rs ([de3803a](https://github.com/MilesCranmer/symbolic_regression.rs/commit/de3803a65eb5b8b7b1892b0ba299ae92b07de98a))


### Code Refactoring

* DRY principles and better hierarchy ([fcce381](https://github.com/MilesCranmer/symbolic_regression.rs/commit/fcce3812e00acb0f55309c878d8f03b9b1064088))
