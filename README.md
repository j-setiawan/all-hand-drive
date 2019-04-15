# All Hand Drive

## Pre-requisites
1. Clone the following repository: [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)
1. `cd tf-pose-estimation`
1. `python setup.py install`
1. Install the packages under `requirements.txt` (if using virtualenv, import global packages to pick up tf-pose-estimation)

## Running
`python all_hand_drive.py [<MQTT host> <MQTT port> <username> <password>]`

Messages are only sent when MQTT information is given.