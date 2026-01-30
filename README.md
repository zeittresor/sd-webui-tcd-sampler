# sd-webui-tcd-sampler

## Table of Contents
- [sd-webui-tcd-sampler](#sd-webui-tcd-sampler)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Install via Command Line](#install-via-command-line)
    - [Install via stable-diffusion-webui](#install-via-stable-diffusion-webui)
    - [Install via HuiShi Launcher (绘世启动器)](#install-via-huishi-launcher-绘世启动器)
  - [Usage](#usage)
  - [Acknowledgements](#acknowledgements)

## Introduction
This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that adds the **TCD** sampling algorithm. It extracts the TCD sampler from the [sd-webui-smea](https://github.com/AG-w/sd-webui-smea) project and makes it available as a standalone extension.

## Installation

### Install via Command Line
Go to the `extensions` directory of your `stable-diffusion-webui` installation:

```bash
cd extensions
```

Clone this repository:

```bash
git clone https://github.com/licyk/sd-webui-tcd-sampler
```

### Install via stable-diffusion-webui
In the `stable-diffusion-webui` interface, go to **Extensions** → **Install from URL**.  
Paste the following into **URL for extension's git repository**:

```
https://github.com/licyk/sd-webui-tcd-sampler
```

Click **Install**.

### Install via HuiShi Launcher (绘世启动器)
Open **HuiShi Launcher (绘世启动器)**, then go to **Version Management** → **Install New Extension**.  
Paste the following into the **Extension URL** field:

```
https://github.com/licyk/sd-webui-tcd-sampler
```

Click **Install** (the button on the right side of the input field).

## Usage
After installation, you can find **TCD** under **Sampling method** in the **txt2img** tab of `stable-diffusion-webui`. Select it to use the TCD sampler.

## Acknowledgements
- [AG-w](https://github.com/AG-w) — for providing the TCD sampling algorithm  
- [ananosleep](https://github.com/ananosleep) — for providing the method to add sampling algorithms to SD WebUI
