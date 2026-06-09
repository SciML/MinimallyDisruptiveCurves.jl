name: CompatHelper

on:
  schedule:
    - cron: '00 * * * *'
  issues:
    types: [opened, reopened]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        julia-version: [1]
        julia-arch: [x86]
        os: [ubuntu-latest]
    steps:
      - uses: julia-actions/setup-julia@latest
        with:
          version: ${{ matrix.julia-version }}
      - name: Pkg.add("CompatHelper")
        run: julia -e 'using Pkg; Pkg.add("CompatHelper")'
      - name: CompatHelper.main()
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: julia -e 'using CompatHelper; CompatHelper.main()'
# name: "Documentation"

# on:
#   push:
#     branches:
#       - master
#     tags: '*'
#   pull_request:

# concurrency:
#   group: ${{ github.workflow }}-${{ github.ref }}
#   cancel-in-progress: ${{ github.ref_name != github.event.repository.default_branch || github.ref != 'refs/tags/v*' }}

# jobs:
#   build-and-deploy-docs:
#     name: "Documentation"
#     uses: "SciML/.github/.github/workflows/documentation.yml@v1"
#     secrets: "inherit"
name: format-check

on:
  push:
    branches:
      - 'master'
      - 'main'
      - 'release-'
    tags: '*'
  pull_request:

jobs:
  runic:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1'
      - uses: fredrikekre/runic-action@v1
        with:
          version: '1'
name: Spell Check

on: [pull_request]

jobs:
  typos-check:
    name: Spell Check with Typos
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Actions Repository
        uses: actions/checkout@v6
      - name: Check spelling
        uses: crate-ci/typos@v1.34.0
name: TagBot
on:
  issue_comment:
    types:
      - created
  workflow_dispatch:
jobs:
  TagBot:
    if: github.event_name == 'workflow_dispatch' || github.actor == 'JuliaTagBot'
    runs-on: ubuntu-latest
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          ssh: ${{ secrets.DOCUMENTER_KEY }}
name: "Tests"

on:
  pull_request:
    branches:
      - master
  push:
    branches:
      - master

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref_name != github.event.repository.default_branch || github.ref != 'refs/tags/v*' }}

jobs:
  tests:
    name: "Tests"
    strategy:
      matrix:
        version:
          - "1"
          - "lts"
          - "pre"
    uses: "SciML/.github/.github/workflows/tests.yml@v1"
    with:
      julia-version: "${{ matrix.version }}"
    secrets: "inherit"

  nopre:
    name: "nopre"
    strategy:
      matrix:
        version:
          - "1"
          - "lts"
    uses: "SciML/.github/.github/workflows/tests.yml@v1"
    with:
      julia-version: "${{ matrix.version }}"
      group: "nopre"
    secrets: "inherit"
