---
icon: lucide/terminal
---

# CLI Tutorial

The `long-toolkit` command-line interface (CLI) is the recommended entry point for new users, since it exposes every tool of the hub without writing code. This tutorial walks through the interface in six short steps, each paired with a video demonstration.

!!! note "Before launching"
    Launch the CLI from inside a Python project in which `long-toolkit` is installed (see [Installation](../getting-started/index.md#installation)), since the CLI calls the Python package through `uv` (the state-of-the-art Python package manager) from the nearest `pyproject.toml`.

## A) Launching the CLI

<div class="cli-tutorial-card">
  <div class="cli-tutorial-card__header">
    <div>Action</div>
    <div>Video Demonstration</div>
  </div>
  <div class="cli-tutorial-card__body">
    <div class="cli-tutorial-card__action">Run <code>long-toolkit</code> from a terminal opened in the Python project. When installed with Homebrew, the binary is already on the <code>PATH</code>.<br><br>Command: <code>long-toolkit</code></div>
    <div class="cli-tutorial-card__media"><video class="cli-tutorial-video" src="../public/resources/tutorials/home.mp4" poster="../../public/resources/tutorials/home.png" controls muted playsinline preload="metadata"></video></div>
  </div>
</div>

## B) Navigating the CLI (Key Strokes)

<div class="cli-tutorial-card">
  <div class="cli-tutorial-card__header">
    <div>Action</div>
    <div>Video Demonstration</div>
  </div>
  <div class="cli-tutorial-card__body">
    <div class="cli-tutorial-card__action">Primary navigation: <code>left/right</code> (or <code>Shift+Tab</code>/<code>Tab</code>) to switch stages, <code>up/down</code> + <code>Enter</code> to select, <code>b</code> to go back, <code>q</code> or <code>Ctrl+C</code> to quit.<br><br>Command palette: <code>:</code> or <code>Ctrl+P</code> to open, <code>Enter</code> to run, <code>Tab</code> autocomplete, <code>Ctrl+H</code> history, <code>Ctrl+L</code> clear, <code>Esc</code> close.</div>
    <div class="cli-tutorial-card__media"><video class="cli-tutorial-video" src="../public/resources/tutorials/navigation.mp4" poster="../../public/resources/tutorials/navigation.png" controls muted playsinline preload="metadata"></video></div>
  </div>
</div>

## C) Understanding the Stages

<div class="cli-tutorial-card">
  <div class="cli-tutorial-card__header">
    <div>Action</div>
    <div>Video Demonstration</div>
  </div>
  <div class="cli-tutorial-card__body">
    <div class="cli-tutorial-card__action">The interface follows the four stages of the hub:<br><br><strong>Stage A, Data Preparation</strong>: convert raw cohort files or generate synthetic longitudinal cohorts.<br><strong>Stage B, Data Preprocessing</strong>: clean, impute, harmonise and reshape repeated measures.<br><strong>Stage C, Machine Learning</strong>: benchmark and train standard and longitudinal models.<br><strong>Stage D, Explainability</strong>: explain a saved model, e.g., the best estimator of a benchmark, with SHAP values.<br><br>Entering a stage opens the list of its tools.</div>
    <div class="cli-tutorial-card__media"><video class="cli-tutorial-video" src="../public/resources/tutorials/understanding-stages.mp4" poster="../../public/resources/tutorials/understanding-stages.png" controls muted playsinline preload="metadata"></video></div>
  </div>
</div>

## D) Generating a Synthetic Cohort

<div class="cli-tutorial-card">
  <div class="cli-tutorial-card__header">
    <div>Action</div>
    <div>Video Demonstration</div>
  </div>
  <div class="cli-tutorial-card__body">
    <div class="cli-tutorial-card__action">Path in the interface:<br><br>1. Open <code>Data Preparation</code>.<br>2. Select <code>Synthetic Data Generation</code>.<br>3. Choose a technique (e.g., <code>trend_patterns</code>).<br>4. Fill the required parameters and run.</div>
    <div class="cli-tutorial-card__media"><video class="cli-tutorial-video" src="../public/resources/tutorials/synthetic-data-gen.mp4" poster="../../public/resources/tutorials/synthetic-data-gen.png" controls muted playsinline preload="metadata"></video></div>
  </div>
</div>

## E) Running Any Tool From the Command Palette

<div class="cli-tutorial-card">
  <div class="cli-tutorial-card__header">
    <div>Action</div>
    <div>Video Demonstration</div>
  </div>
  <div class="cli-tutorial-card__body">
    <div class="cli-tutorial-card__action">Open the palette with <code>:</code> or <code>Ctrl+P</code>, type the name of a tool (e.g., <code>benchmark longitudinal ml</code>), and press <code>Enter</code>. This avoids navigating menu by menu.</div>
    <div class="cli-tutorial-card__media"><video class="cli-tutorial-video" src="../public/resources/tutorials/command-palette.mp4" poster="../../public/resources/tutorials/command-palette.png" controls muted playsinline preload="metadata"></video></div>
  </div>
</div>

## F) Quitting the CLI

<div class="cli-tutorial-card">
  <div class="cli-tutorial-card__header">
    <div>Action</div>
    <div>Video Demonstration</div>
  </div>
  <div class="cli-tutorial-card__body">
    <div class="cli-tutorial-card__action">From the navigator, press <code>q</code> (or <code>Ctrl+C</code>) to quit. From inside a tool, go back to the navigator first, then quit.</div>
    <div class="cli-tutorial-card__media"><video class="cli-tutorial-video" src="../public/resources/tutorials/quit-cli.mp4" poster="../../public/resources/tutorials/quit-cli.png" controls muted playsinline preload="metadata"></video></div>
  </div>
</div>
