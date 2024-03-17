# Templates: Reinforcement Learning
A place to store examples of reinforcement learning code


# Configuring and Environment

To get the gymnasium library to install correctly these days, you often have to do some non-pip installs in your environment. The following instructions have worked well for WSL2 with Ubuntu. 

## Using Anaconda

1. Create your virtual environment
    
    ```conda create -n <env_name> python==3.11```

2. Activate your virtual environment

    ```conda activate <env_name>```

3. Install the requirements with pip

    ```pip install -r requirements.txt```

4. Deactivate when you're done

    ```conda deactivate```

## Using Python Virtual Environment

1. Create your virtual environment

    ```python -m venv path/to/env/env_name```

2. Activate your virtual environment

    ```source path/to/env/env_name/bin/activate```

3. Upgrade your pip to the latest version

    ```pip install --upgrade pip```

4. Install the requirements with pip

    ```pip install -r requirements.txt```

4. Deactivate when you're done

    ```deactivate```


## Install Dependencies Via Linux ([source](https://learn.microsoft.com/en-us/cpp/build/walkthrough-build-debug-wsl2?view=msvc-170))

```
sudo apt update
sudo apt upgrade
sudo apt install g++ gdb make ninja-build rsync zip
```

## Install Dependencies Via Pip

```
pip install swig
pip install -r requirements.txt
```



