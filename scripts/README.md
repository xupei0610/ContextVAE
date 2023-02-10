## Data Preprocessing

### nuScenes

#### Create Environment

    $ conda create -n nuscenes
    $ conda activate nuscenes
    $ conda install python=3.9
    $ pip install nuscenes-devkit==1.1.9
    $ pip install pyquaternion==0.9.9
    $ pip install pandas==1.3.4

#### Prepare data
Download `Full dataset (v1.0)` - `Trainval` - `Metadata` and `Map expansion pack (v1.3)` from https://www.nuscenes.org/nuscenes#download

Uncompress `v1.0-trainval_meta.tgz` into `<raw_dataset_folder>` folder.

Uncompress `nuScenes-map-expansion-v1.3.zip` into `<raw_dataset_folder>/map` folder.


#### Preprocessing
Run command

    $ python nuscenes.py <raw_dataset_folder> <target_dataset_folder>

It will put training data into the folder of `<target_dataset_folder>/train`, evaluation data into `<target_dataset_folder>/val`, and map data into `<target_dataset_folder>/map`.


### Lyft

#### Create Environment

    $ conda create -n lyft
    $ conda activate lyft
    $ conda install python=3.9
    $ pip install l5kit==1.5.0
    $ pip install opencv-python

#### Prepare data

Download dataset

    $ wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/train.tar
    $ wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/validate.tar
    $ wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/semantic_map.tar

Uncompress downloaded data to `<raw_dataset_folder>`

    $ tar -xf train.tar -C <raw_dataset_folder>/train
    $ tar -xf validate.tar -C <raw_dataset_folder>/validate
    $ tar -xf semantic_map.tar -C <raw_dataset_folder>/semantic_map


#### Preprocessing
Run command

    $ python lyft.py <raw_dataset_folder> <target_dataset_folder> --frameskip 2 --split 4

It will split the data into 4 parts, and put training data into the folder of `<target_dataset_folder>/train`, evaluation data into `<target_dataset_folder>/validate` and map data into `<target_dataset_folder>/map`.


### Waymo

#### Create Environment

    $ conda create -n waymo
    $ conda activate waymo
    $ conda install python=3.9
    $ conda install -c conda-forge openexr-python
    $ conda install -c conda-forge gsutil
    $ pip install waymo-open-dataset-tf-2-6-0
    $ pip install opencv-python

#### Prepare data

Download data to `<raw_dataset_folder>`

    $ gsutil -m cp -r \
        "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training" \
        "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/validation" \
        <raw_dataset_folder>

#### Preprocessing
Run command

    $ python waymo.py <raw_dataset_folder> <target_dataset_folder> --frameskip 2 --split 8


It will split the data into 8 parts, and put training data into the folder of `<target_dataset_folder>/training`, evaluation data into `<target_dataset_folder>/validation` and map data into `<target_dataset_folder>/map`.
