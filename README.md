# <img src="images/nRootTag-round.png" alt="nRootTag application icon" height=24 width=24 valign=bottom/> nRootTag
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Languages](https://img.shields.io/badge/Written%20with-CUDA%2FC%2B%2B%2FPython-green)
![DockerImage](https://img.shields.io/badge/Docker-KeySearch-blue?style=flat&link=https%3A%2F%2Fhub.docker.com%2Frepository%2Fdocker%2Fchiba765%2Fnroottag-seeker)
![ArtifactRainbowTable](https://img.shields.io/badge/Artifact-RainbowTable-orange)
![Paper](https://img.shields.io/badge/To%20be%20in%20-%20USENIX%20Security%20'25%20-red?link=https%3A%2F%2Fcs.gmu.edu%2F~zeng%2Fpapers%2F2025-security-nrootgag.pdf)


Our work uncovered a vulnerability in the Find My service that permitted all types of BLE addresses for advertising. Leveraging this flaw, we proposed a novel attack method, **nRootTag**, which transformed a computer into an ''AirTag'' tracker without requiring root privilege escalation.


# Evaluation

The project forms a complete attack chain and depends on each component working together. The setup might be sophisticated, we thank you for your patience. The project contains the following components: **C&C Server**, **Database**, **Seeker**, and **Trojans** for Linux, Windows, and Android, respectively. Each component can be evaluated separately.

📺 We provide screen recordings for essential steps. Due to size constrain of GitHub, please download the screen recordings from [Zenodo](https://doi.org/10.5281/zenodo.14728530). They are available under **ScreenRecording** directory. Please review [Evaluation.md](Evaluation.md) for detailed steps to reproduce and evaluate our project.


# Find My Report Retrieval

We created [Chapoly1305/FindMy](https://github.com/Chapoly1305/FindMy) for our experiment. You may also visit other existed projects on the Internet to retrieve and develop your own retrieval platform. We do not endorse or vouch for any of these projects.

# Responsible Disclosure & Advisory

We have contacted Apple regarding the vulnerability and attack method. Apple has [acknowledged](https://support.apple.com/en-us/121837#:~:text=for%20their%20assistance.-,Proximity,-We%20would%20like) the issue and implementing fix. This code is for academic research and security analysis only. Use responsibly in controlled test environments.


# Research Paper

Please consider sharing and citing our research paper [*Tracking You from a Thousand Miles Away! Turning a Bluetooth Device into an Apple AirTag Without Root Privileges*](https://cs.gmu.edu/~zeng/papers/2025-security-nrootgag.pdf)!


```
@inproceedings{chen2025track,
title={Tracking You from a Thousand Miles Away! Turning a Bluetooth Device into an Apple AirTag Without Root Privileges},
author={Chen, Junming and Ma, Xiaoyue and Luo, Lannan and Zeng, Qiang},
booktitle={USENIX Security Symposium (USENIX Security)},
year={2025}
}
```

# License and Credits

nRootTag uses GPL v3, inherits the license from the original projects. We appreciate the authors for their contributions.

- [OpenHayStack](https://github.com/seemoo-lab/openhaystack) - GPL v3
- [VanitySearch](https://github.com/JeanLucPons/VanitySearch) - GPL v3
- [win-ble-cpp](https://github.com/urish/win-ble-cpp) - MIT
- [Windows-universal-samples](https://github.com/microsoft/Windows-universal-samples) - MIT

