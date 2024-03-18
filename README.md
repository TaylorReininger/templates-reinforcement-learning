# Templates: Reinforcement Learning
A place to store examples of reinforcement learning code.


# Configuring and Environment

It is always recommended to do your work in virtual environments. To use Anaconda or Python Virtualenv, simply follow the instructions here. This particular dependency list can be challenging to configure due to some of the changes when switching from the original OpenAI Gym to the newer Farama [Gymnasium](https://gymnasium.farama.org/) or [PettingZoo](https://pettingzoo.farama.org/) . I have done my best to document several working solutions, but things may also break again in the future. At the moment, I am having success on both Windows and Linux with ```gymnasium == 0.29.1``` and ```box2d-py == 2.3.5```.

## Using Anaconda

1. Create your virtual environment
    
    ```conda create -n <env_name> python==3.11```

2. Activate your virtual environment

    ```conda activate <env_name>```

3. Install box2d with Anaconda since this will give pip issues

    ```conda install -c conda-forge box2d-py```

3. Install the requirements with pip

    ```pip install swig```

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


## Troubleshooting

For some reason, the box2d library frequently gives me issues at installation time. If you are getting errors with this part, there are some other dependencies that may need to be installed.  






To get the gymnasium library to install correctly, it is often necessary to do some non-pip installs in your environment. The following instructions have worked well for WSL2 with Ubuntu. 


### Install Dependencies Via Linux ([Source](https://learn.microsoft.com/en-us/cpp/build/walkthrough-build-debug-wsl2?view=msvc-170))

```
sudo apt update
sudo apt upgrade
sudo apt install g++ gdb make ninja-build rsync zip
```
Then proceed with the instructions above again. 



