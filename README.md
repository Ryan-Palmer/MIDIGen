# MIDIGen Next Note Prediction

A series of Jupyter notebooks exploring MIDI next-note prediction as a way of learning about topics such as tokenisation and transformer architectures.

It takes heavy inspiration from Andrew Shaw's [MusicAutobot](https://github.com/bearpelican/musicautobot/tree/master) project, Andrej Karpathy's [Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series and Nick Ryan's [Coding a Paper](https://www.youtube.com/playlist?list=PLam9sigHPGwOe8VDoS_6VT4jjlgs9Uepb) series.

It is set up as a dev container so the whole environment can be launched with a single command and connected to using VSCode (or through the browser).

## Requirements
To run on Windows you will need:

- An NVidia GPU
- Docker Desktop
- WSL2
- VSCode with the DevContainers extension

## Adding dependencies

### Ubuntu
If you need to `apt-get` any apps, add them to the `Dockerfile` where indicated.

### Python
Add any pip packages you need to `pip-packages.txt` before building the container.

## Build and Run
To build and launch the VSCode and Jupyter servers in a dev container, open the command pallete and run `Dev Containers: Reopen in Container`.

Once it has finished launching then VSCode should auto-select the Jupyter Python kernel and prompt you for the Jupyter password, which is `dev`.

> This password can be changed in the Docker file or ommitted completely to force Jupyter to create a new GUID key which be displayed along with the server URL in the terminal.

If it doesn't do this automatically, set the kernel of your Polyglot Notebooks by 
- Clicking the kernel selector at the top right of the notebook.
- Selecting `Existing Jupyter Server`.
- Scroll down and select `Enter the URL of the running Jupyter Server`.
- Paste in `http://127.0.0.1:8888/lab?token=dev`
- Press enter again to select `127.0.0.1` (which will be prefilled)
- Select `Python 3 (ipykernel)`.

You can also visit that URL in your browser if you would rather use the Jupyter Labs IDE.

> If you see some errors about the .NET SDK version and Python Kernel being invalid after first launch, don't worry, the installer is just catching up. You don't need to install anything.

## Workspace files
Any files you put in the `src` folder will be mounted and available to the Docker container / Jupyter server.

### Adding VSCode extensions
After you launch the DevContainer you can select extensions in VSCode, click their settings cog and select `Add to devcontainer.json`.

Once you have done this for all the extensions you need, open the command pallete and run `Dev Containers: Rebuild and Reopen in Container`.
