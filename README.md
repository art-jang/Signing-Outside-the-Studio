# Signing Outside the Studio: Benchmarking Background Robust Continuous Sign Language Recognition

This code reproduces Scene-PHOENIX benchmark dataset via our automatically generating algorithm using scene database LSUN and SUN397.
- For any details about the algorithm, please refer to the paper and the supplementary materials. 
- Note that due to any potential copyright issues, we do not re-distribute the existing benchmarks whether or not they are modified with backgrounds. 


## Requirements
- Human Segmentation Model
    - Follow the instructions to setup the segmentation model.
    - https://github.com/thuyngch/Human-Segmentation-PyTorch
    - Download the pre-trained *UNet_MobileNetV2 (alpha=1.0, expansion=6)*  (see the repository) to the location:  `Human-Segmentation-PyTorch/pretrained`.

- LSUN Database
    - Follow the instructions to download the LSUN database to `{DATA_PATH}/lsun` (ex: data/lsun)
    - https://github.com/fyu/lsun

- SUN397
    - Download SUN397 **Image Database** and **Partition** from the following link.
    - https://vision.princeton.edu/projects/2010/SUN/
    - Unzip the `Partitions.zip` to `{DATA_PATH}/SUN397/Partitions`.

- PHOENIX-2014
    - Download *RWTH-PHOENIX-Weather 2014: Continuous Sign Language Recognition* Dataset to `{DATA_PATH}/phoenix2014-release`
    - https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/


## Step-by-step to generate Scene-PHOENIX
1. Locate `bg_dataset.py` and `generate_scene_phoenix.py` on `Human-Segmentation-PyTorch` folder.
2. Copy the attached `lsun` and `SUN397` folder in the code to your dataset path `{DATA_PATH}`.
    - The txt files therein specifies the index of each data to be used for synthesizing Scene-PHOENIX with background. 
    - Note that although the pre-defined indices are the identical in SUN397, we use different test partitions provided from the SUN397 authors, which are specified as `Partitions`. 

3. Run the `generate_scene_phoenix.py`.
    ```
    python main.py  --sign_root {PATH_TO_PHOENIX} \
                --sign_split dev  \   # dev or test
                --background_type \   # LSUN or SUN397
                --partition 1     \   # Different partitions (1, 2, 3)
    ```

    - Note that the variable `bg_root` in `generate_scene_phoenix.py` should be modified to your paths of LSUN and SUN397 data.

4. The Scene-PHOENIX benchmark datasets are created on the locations of PHOENIX-2014 dataset.
