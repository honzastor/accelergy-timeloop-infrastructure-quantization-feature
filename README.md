# Native Installation of Timeloop+Accelergy (and optionally GAMMA)

## Timeloop+Accelergy framework (with added quantization support)
The infrastructure is composed of:
* **Timeloop** – infrastructure that aims to provide modeling, mapping and code-generation for Explicitly-Decoupled Data Orchestration (EDDO) architectures, with a focus on for dense- and sparse- tensor algebra workloads.
* **Accelergy** – a tool to provide the energy and area characteristics of a variety of compound components that make up a design.

Useful links:
* #### [Timeloop paper](https://ieeexplore.ieee.org/document/8695666)
* #### [Accelergy paper](https://ieeexplore.ieee.org/document/8942149)
* #### [Sparseloop paper](https://ieeexplore.ieee.org/document/9923807)
* #### [Documentation](https://timeloop.csail.mit.edu/)
* #### [Tutorials](https://accelergy.mit.edu/tutorial.html) 
* #### [Official GitHub](https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure)

To work with the updated Timeloop+Accelergy locally, you must first download the infrastructure containing the quantization feature update, then build it and export PATHs to the compiled binary files to allow execution from any location.

1. Clone the repository:
```bash
mkdir timeloop-accelergy
cd timeloop-accelergy
git clone --recurse-submodules https://github.com/honzastor/accelergy-timeloop-infrastructure-quantization-feature.git
```

2. Install it. For detailed installation guide (continue after the git clone command) along with the list of required dependencies see [here](https://accelergy.mit.edu/infra_instructions.html)

3. Test it. Once it is done, you can check successful installation by running the binaries from any location, i.e.:
```bash
accelergy
accelergyTables

timeloop-mapper
timeloop-model
timeloop-metrics
```

## GAMMA for Timeloop (optional)

**GAMMA** (Genetic Algorithm-based Mapper for ML Accelerators) is a search heuristic that automates the HW mapping of DNN models on accelerators via genetic algorithm.

Useful links:
#### [GAMMA paper](https://ieeexplore.ieee.org/document/9256431)
#### [GitHub](https://github.com/maestro-project/gamma-timeloop.git)

GAMMA implementation for Timeloop as a cost model utilizes PyTimeloop, a Python interface for Timeloop, that is included in the Timeloop+Accelergy framework. To build PyTimeloop, one must first install the infrastructure itself and then proceed with the package installation steps provided within `README.md` inside `timeloop-python` directory (located in `src` directory of the infrastructure downloaded in previous step). Optionally you can use a script that comes with GAMMA.

I **highly recommend** using the `build_pytimeloop.py` script downloaded within the GAMMA for Timeloop repository. It not only clones and updates the timeloop-python repository into your desired location, but most importantly, it applies a rollback patch to a more stable version. Sadly the current version of PyTimeloop's package build is accompanied by errors (from personal experience, may be resolved in the future; though you can try your luck and install it without the patch).

If you encounter any problems, check the attached `README.md` inside the `timeloop-python` for more information (you may need to export some PATHs, etc.).

1. Clone the repository:
```bash
mkdir timeloop-gamma 
git clone https://github.com/maestro-project/gamma-timeloop.git timeloop-gamma
```

2.  Build PyTimeloop for GAMMA to use:
```bash
python build_pytimeloop.py
```
Note: You may have to modify the `pytimeloop_dir` path inside the script to your timeloop-python preferred location.
(Optionally follow the steps inside README.md in timeloop-python.)

3. Test a successful build by running:
```bash
./run_gamma_timeloop.sh
```

Start the container – NOTE(!) that this is the original instructions (not tested for the modified repository)
--------------------------------------------------------------------

- Put the *docker-compose.yaml* file in an otherwise empty directory
- Cd to the directory containing the file
- Edit USER_UID and USER_GID in the file to the desired owner of your files (echo $UID, echo $GID)
- Run the following command:
```
      % If you are using x86 CPU (Intel, AMD)
      % DOCKER_ARCH=amd64 docker-compose run infrastructure 

      % If you are using arm CPU (Apple M1/M2)
      % DOCKER_ARCH=arm64 docker-compose run infrastructure 

      % If you want to avoid typing "DOCKER_ARCH=" every time,
      % "export DOCKER_ARCH=<your architecture>" >> ~/.bashrc && source ~/.bashrc
```
- Follow the instructions in the REAME directory to get public examples for this infrastructure


Refresh the container
----------------------

To update the Docker container run:

```
     % If you are using x86 CPU (Intel, AMD)
     % DOCKER_ARCH=amd64 docker-compose pull 

     % If you are using arm CPU (Apple M1/M2)
     % DOCKER_ARCH=arm64 docker-compose pull

     % If you want to avoid typing "DOCKER_ARCH=" every time,
     % "export DOCKER_ARCH=<your architecture>" >> ~/.bashrc && source ~/.bashrc
````

Build the image
---------------

```
      % git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
      % cd accelergy-timeloop-infrastructure
      % export DOCKER_EXE=<name of docker program, e.g., docker>
      % make pull
      % "make build-amd64" or "make build-arm64" depending on your architecture
```

Push the image to docker hub
----------------------------

```
      % cd accelergy-timeloop-infrastructure
      % export DOCKER_NAME=<name of user with push privileges>
      % export DOCKER_PASS=<password of user with push privileges>
      % "make push-amd64" or "make push-arm64"
```
