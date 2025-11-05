
<h1 align="center">TumorMap: A Laser-based Surgical Platform for 3D Tumor Mapping and Fully-Automated Tumor Resection</h1>

<center>
Guangshen Ma, PhD,1,2 †, * Ravi Prakash,1, †, * Beatrice Schleupner,3 Jeffrey Everitt, DVM,4 Arpit Mishra, PhD,1
Junqin Chen, PhD, 1 Brian Mann, PhD, 1 Boyuan Chen, PhD, 1 Leila Bridgeman, PhD, 1 Pei Zhong, PhD, 1
Mark Draelos, MD, PhD, 2,5 William C. Eward, DVM, MD 3 and Patrick J. Codd, MD1,6, † 
</center>

<br>

<!-- <div style="text-align:center;"> -->
<center>
<br>
G. Ma and R. Prakash contributed equally to this work.
<br>
Corresponding email: guangshe@umich.edu, ravi.prakash@duke.edu, patrick.codd@duke.edu.
<br>
1. Thomas Lord Department of Mechanical Engineering and Materials Science, Duke University
<br>
2. Department of Robotics, University of Michigan, Ann Arbor
<br>
3. Department of Orthopaedic Surgery, School of Medicine, Duke University
<br>
4. Department of Pathology, School of Medicine, Duke University
<br>
5. Department of Ophthalmology and Visual Sciences, University of Michigan Medical School, Ann Arbor
<br>
6. Department of Neurosurgery, School of Medicine, Duke University
</center>
<!-- </div> -->

## Overview
Surgical resection of malignant solid tumors is critically dependent on the surgeon’s ability to accurately identify pathological
tissue and remove the tumor while preserving surrounding healthy structures. However, building an intraoperative 3D tumor model
for subsequent removal faces major challenges due to the lack of high-fidelity tumor reconstruction, difficulties in developing
generalized tissue models to handle the inherent complexities of tumor diagnosis, and the natural physical limitations of bimanual
operation, physiologic tremor, and fatigue creep during surgery. To overcome these challenges, we introduce “TumorMap", a
surgical robotic platform to formulate intraoperative 3D tumor boundaries and achieve autonomous tissue resection using a set of
multifunctional lasers. TumorMap integrates a three-laser mechanism (optical coherence tomography, laser-induced endogenous
fluorescence, and cutting laser scalpel) combined with deep learning models to achieve fully-automated and noncontact tumor
resection. We validated TumorMap in murine osteoscarcoma and soft-tissue sarcoma tumor models, and established a novel
histopathological workflow to estimate sensor performance. With submillimeter laser resection accuracy, we demonstrated
multimodal sensor-guided autonomous tumor surgery without any human intervention.
<!-- <div style="text-align: center;">
  <img src="./DelayKoop/make_gif_high_res.gif" alt="make_gif_high_res" width="600">
</div> -->

<!-- <br> -->
<!-- Duke University -->

<!-- # tumormap_codebase_main -->
<!-- tumormap codebase for review -->

## Note:  
    The matlab code should be ready for run locally. Add the entire folder in the system path to make sure the repo can be run correctly with the utility functions. All the supporting and utility functions are within this code repository. 
    The python packages can be installed based on each script of the codes (refer to the packages descriptions at the beginning of each script). After that, the python code should be ready to run locally. 

## Contents
- [Module-1: MATLAB (Tumor Mapping)](#prerequisites)
- [Module-2: Python (Robot Planning)](#training)
<!-- - [Logging](#logging)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements) -->

## Module-1: MATLAB (Tumor Mapping)

    System calibration:
        Script-1: calib_oct_to_cam_extrinsics_revisit.m. This script summarizes the camera intrinsics calibration.
        Script-2: calib_oct_to_cam_intrinsics_revisit.m. This script summarizes the OCT-to-camera extrinsics calibration.
        Script-3: calib_oct_to_laser_axis_revisit.m. This script summarizes the OCT-to-laser-axis calibration.
        Script-4: calib_oct_to_laser_dir_revisit.m. This script summarizes the OCT-to-laser-orientation calibration.
        Script-5: calib_vis_oct_to_laser.m. This script is used to visualize the OCT-to-laser calibration configuration. 
        Script-6: calib_vis_tform.m. This script is used to visualize the transformation relation between the sensor components. 
    System ,odeling: Kinematics and planning.
        Script-1: model_ik_vis.m. This script shows how to use the inverse kinematics solver to find the optimal laser configurations to visit the target point at the tissue surface. 
        Script-2: model_laser_trajectory.m. This script shows an integration visualization of the laser trajectory to trace the tissue targets.
        Script-3: model_tumor_geometry.m. This script shows how to segment the tumor boundary from the predicted tumor tags and formulate the 3D tumor map. 
    Ex vivo tissue experiment:
        Script-1: exvivo_exp_diode.m. This script shows the ex vivo tissue experiment by using the visble laser diode scalpel.
        Script-2: exvivo_exp_fiber.m. This script shows the ex vivo tissue experiment by using the fiber-coupled laser scalpel.
    Mice tumor experiment: 
        Script-1: mice_exp_os.m. This script shows the osteosarcoma (OS) tumor experiment using the fiber-coupled laser scalpel.
        Script-2: mice_exp_sts.m. This script shows the soft tissue sarcoma (STS) tumor experiment using the fiber-coupled laser scalpel. 

## Module-2: Python (Robot Planning and Main Experiment Workflow)

    Experiment and workflow: 
        Script-1: exp_robot_planning.py. This script summarizes the robot planning and simulation methods to prepare for the actual experiments.
        Script-2: exp_workflow.py. This script summarizes the workflows to achieve each utility and functions for the actual experiments.
        Script-3: exp_system_integration.py. This script provides a solution of how to control the robot arm and sensors jointly to perform experiments. This is an example solution, since the actual robot and sensor systems will depend on the specialized experimental setup (for different users and applications).
    Machine learning model training and inference (prediction):
        Script-1: mlp_train_and_inference_cross_validation_tumor_exp.py. This script summarizes the entire model training, testing and inference for the tumor resection epxeriments.
        Script-2: mlp_train_and_inference_cross_validation_tumor_histopathology.py. This script summarizes the entire model training, testing and inference for the histopathology pxeriments.
        Script-3: mlp_inference_main.py. This script provides a solution to show how to perform model inference. 