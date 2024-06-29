# Pytorch DevContainer template

To run on Windows you will need:

- An Nvidia GPU with latest drivers installed
- Docker Desktop
- WSL2
- VSCode with the DevContainers extension

## Files
Any files you put in the `src` folder will be available to the Docker container and the Jupyter server.

## Dependencies

### Ubuntu
If you need to `apt-get` any apps, add them to the Dockerfile where indicated.

### Pip
Add any pip packages you need to pip-packages.txt before building the container.

## Build and Run
To build and launch the VSCode and Jupyter servers in a dev container, open the command pallete and run `Dev Containers: Reopen in Container`.

Once it has finished launching then VSCode should auto-select the Jupyter Python kernel and prompt you for the Jupyter password, which is `dev`.

> This can be changed in the Docker file or ommitted completely to force Jupyter to create a new GUID key which be displayed along with the server URL in the terminal.

If it doesn't do this automatically, set the kernel of your Polyglot Notebooks session to `http://127.0.0.1:8888/lab?token=dev`.

### VSCode extensions
After you launch the DevContainer (see below) you can select extensions in VSCode, click their settings cog and select "Add to `devcontainer.json`".

Once you have done this for all the extensions you need, open the command pallete and run `Dev Containers: Rebuild and Reopen in Container`.