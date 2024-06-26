import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import os
import yaml
from tqdm import tqdm

from phc.utils import torch_utils
import joblib
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import gc
from scipy.spatial.transform import Rotation as sRot
import random
from phc.utils.flags import flags
from enum import Enum
USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

# COMPLETED ##################################################################################################### 


cartwheels = [
'0-Eyes_Japan_Dataset_takiguchi_turn-04-cartwheels-takiguchi_poses', 
'0-Eyes_Japan_Dataset_hamada_turn-04-cartwheels-hamada_poses', 
'0-ACCAD_Male2MartialArtsExtended_c3d_Extended 3_poses', # capoaera + cartwheel 
'0-BioMotionLab_NTroje_rub009_0030_scamper1_poses', # cartwheel into weird kicks 
'0-BioMotionLab_NTroje_rub011_0030_scamper_poses', # carthweels, bridges, sit on the ground, into bridge 
'0-BioMotionLab_NTroje_rub087_0030_scamper_poses', #spin in circles into cartwheel 
'0-CMU_88_88_07_poses',  # cartwheel 
] # only last 2 succeeded '0-BioMotionLab_NTroje_rub087_0030_scamper_poses' '0-CMU_88_88_07_poses'

# cartwheels_failed2_subset1=['0-ACCAD_Male2MartialArtsExtended_c3d_Extended 3_poses', 
#                             '0-BioMotionLab_NTroje_rub009_0030_scamper1_poses', ] # all failed 

cartwheels_failed2_subset2 =['0-BioMotionLab_NTroje_rub011_0030_scamper_poses', 
                            '0-Eyes_Japan_Dataset_hamada_turn-04-cartwheels-hamada_poses', 
                            '0-Eyes_Japan_Dataset_takiguchi_turn-04-cartwheels-takiguchi_poses'] # last one success 

cartwheels_failed3_subset1 = ['0-ACCAD_Male2MartialArtsExtended_c3d_Extended 3_poses'] # good!
cartwheels_failed3_subset2 = ['0-BioMotionLab_NTroje_rub009_0030_scamper1_poses'] # good "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset2/Humanoid.pth"

cartwheels_failed3_subset3 = ['0-BioMotionLab_NTroje_rub011_0030_scamper_poses'] # good 

cartwheels_failed3_subset4 = ['0-Eyes_Japan_Dataset_hamada_turn-04-cartwheels-hamada_poses'] # good

handstands = [
'0-KIT_200_Handstand01_poses', 
'0-KIT_200_Handstand04_poses', 
'0-KIT_200_Handstand02_poses',
'0-KIT_200_Handstand05_poses', 
'0-KIT_200_Handstand06_poses',
'0-Eyes_Japan_Dataset_aita_pose-16-handstand-aita_poses', 
'0-Eyes_Japan_Dataset_takiguchi_pose-16-handstand-takiguchi_poses', 
'0-Eyes_Japan_Dataset_shiono_pose-16-handstand-shiono_poses',
'0-MPI_Limits_03099_op4_poses', #handstand 
'0-MPI_Limits_03099_op5_poses', #handstand with splits 
# '0-CMU_85_85_05_poses', #handstand with kicks 
# '0-CMU_85_85_12_poses', #shuffling with single handstand 
# '0-CMU_85_85_13_poses', #handstand with falling action to recovery 
]
handstands_failed2_subset1 = ['0-CMU_85_85_05_poses'] # good! 
handstands_failed2_subset2 = ['0-CMU_85_85_12_poses', '0-CMU_85_85_13_poses'] # FAILED  

handstands_failed3_subset1 = ['0-CMU_85_85_12_poses'] # good! /move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed3_subset1_retry/Humanoid.pth
handstands_failed3_subset2 = ['0-CMU_85_85_13_poses'] # good! 

# RUNNING
ground2getup = [ 
'0-BMLmovi_Subject_71_F_MoSh_Subject_71_F_17_poses', #sit on the ground and row, then stand up 
'0-CMU_111_111_08_poses', # ground to get up 
# '0-CMU_139_139_18_poses', # ground to get up 
'0-CMU_140_140_04_poses', # ground to get up
'0-CMU_140_140_08_poses',  # ground to get up
'0-CMU_140_140_09_poses',  # ground to get up
'0-CMU_114_114_11_poses', #floor to get up and walk 
# '0-Eyes_Japan_Dataset_shiono_accident-04-damage right leg-shiono_poses', # limp then ground to get up 
# '0-TotalCapture_s4_freestyle3_poses', # stumbling forwards, then get up from the ground 
# '0-MPI_Limits_03099_op3_poses', # bridge into get up 
'0-SFU_0018_0018_Bridge001_poses', #bend backward into bridge with leg raise 
# '0-TotalCapture_s5_freestyle3_poses',  # run around, throw objects, get on the ground then get up 
'0-CMU_77_77_18_poses', #get up off the ground
] # /move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup
ground2getup_failed2_subset1 = ['0-TotalCapture_s5_freestyle3_poses',  '0-MPI_Limits_03099_op3_poses', '0-CMU_139_139_18_poses'] # /move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed2_subset1/Humanoid.pth
ground2getup_failed2_subset2 = ['0-TotalCapture_s4_freestyle3_poses' ,'0-Eyes_Japan_Dataset_shiono_accident-04-damage right leg-shiono_poses']  # /move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed2_subset2/Humanoid.pth
ground2getup_failed3_subset1 = ['0-MPI_Limits_03099_op3_poses'] # "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed3_subset1/Humanoid.pth"



back_flips = [
'0-CMU_90_90_19_poses', # crawl forward to backflip 
'0-CMU_90_90_28_poses', # ground twirl (maybe with around the world? other) 
# '0-CMU_90_90_29_poses',  #cartwheel into donkey kicks into backflip 
# '0-CMU_85_85_14_poses', #backflip with one arm handstand 
# '0-CMU_88_88_05_poses', # cartwheel into backflip 
'0-CMU_88_88_08_poses', #backflip 
'0-CMU_88_88_09_poses', #backflip 
'0-CMU_90_90_08_poses', #sideflip 
'0-CMU_85_85_01_poses', #backflip 
'0-CMU_85_85_02_poses', #backflip 
]
backflips_failed2_subset1 = ['0-CMU_85_85_14_poses'] #  good! len 743 
backflips_failed2_subset2 = ['0-CMU_90_90_29_poses'] # good! 
backflips_failed2_subset3 = ['0-CMU_88_88_05_poses'] # good



roll = [
'0-CMU_90_90_33_poses',  #roll on ground forward
'0-CMU_90_90_34_poses',  #roll on ground forward
'0-CMU_90_90_35_poses',  #roll on ground forward
'0-CMU_90_90_36_poses',  #roll on ground forward
'0-CMU_127_127_23_poses', # roll forward (tumble)
'0-CMU_127_127_24_poses', # roll forward (tumble)
'0-CMU_128_128_10_poses', # roll forward (tumble)
'0-CMU_128_128_11_poses', # roll forward (tumble)
'0-CMU_90_90_01_poses',  #rolling backwards on the ground 
# '0-SFU_0017_0017_ParkourRoll001_poses', # roll forwards
] # Good!  

break_dance = [
'0-CMU_85_85_08_poses', #around the world with legs
'0-CMU_85_85_04_poses', #break dance with single hand up (prob train seperately)
] # good! 

front_flips =[ 
'0-CMU_85_85_06_poses', #front flip 
'0-CMU_85_85_07_poses', #front flip 
] # good! 

floor_poses = [
'0-ACCAD_Male1General_c3d_General A8 - Crouch to Lie Down_poses', 
# '0-SFU_0007_0007_Crawling001_poses', 
'0-ACCAD_Male2General_c3d_A12- Crawl Backwards_poses', # army crawl backwards
] # good! 

sprints=  [
'0-ACCAD_Male2Running_c3d_C20 - run to pickup box_poses', #sprint forwards pickup box 
# '0-MPI_HDM05_tr_HDM_tr_05-03_01_120_poses', # limp into sprints into cartwheels 
# '0-MPI_HDM05_tr_HDM_tr_05-03_02_120_poses', # limp into sprints into cartwheels 
# '0-MPI_HDM05_tr_HDM_tr_05-03_03_120_poses', # limp into sprints into cartwheels 
# '0-MPI_HDM05_dg_HDM_dg_05-03_01_120_poses', # limping into sudden sprints into cartwheel 
# '0-MPI_HDM05_dg_HDM_dg_05-03_02_120_poses', # sudden sprints 
# 0-MPI_HDM05_mm_HDM_mm_05-03_02_120_poses', # lurch forward and limp, sudden sprints, carrtwheel 
]



kicks_punches = [
# '0-MPI_HDM05_tr_HDM_tr_03-02_04_120_poses', # kicks and punches 
# '0-MPI_HDM05_tr_HDM_tr_05-01_03_120_poses', # kicks and punches 
'0-ACCAD_Male2MartialArtsKicks_c3d_G7 -  capoera_poses', #spin kick 
# '0-BioMotionLab_NTroje_rub048_0030_scamper_poses' # sided leg raises into kicks in all directions into balance on one foot onto ground then get up 
]

weird_walks = [
# '0-BioMotionLab_NTroje_rub010_0029_scamper_poses', # bizare walks into crab walks backwards into rolling on the ground
'0-BioMotionLab_NTroje_rub012_0031_scamper_poses', # obscure walks 
]
scamper_failed2 = ['0-BioMotionLab_NTroje_rub048_0030_scamper_poses', '0-BioMotionLab_NTroje_rub010_0029_scamper_poses'] # need to rerun 
scamper_failed3 = ['0-BioMotionLab_NTroje_rub048_0030_scamper_poses']


policy2motion = {
    # groun2getup
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed3_subset1/Humanoid.pth": ['0-MPI_Limits_03099_op3_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed2_subset2/Humanoid.pth":  ['0-TotalCapture_s4_freestyle3_poses' ,'0-Eyes_Japan_Dataset_shiono_accident-04-damage right leg-shiono_poses'], 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed2_subset1/Humanoid.pth": ['0-TotalCapture_s5_freestyle3_poses',  '0-CMU_139_139_18_poses'], 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup/Humanoid.pth": ['0-BMLmovi_Subject_71_F_MoSh_Subject_71_F_17_poses', '0-CMU_111_111_08_poses', '0-CMU_140_140_04_poses', '0-CMU_140_140_08_poses', '0-CMU_140_140_09_poses', '0-CMU_114_114_11_poses', '0-SFU_0018_0018_Bridge001_poses', '0-CMU_77_77_18_poses'],
    
    # handstands 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstand_failed3_subset2/Humanoid.pth": ['0-CMU_85_85_13_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstands_failed3_subset1_retry/Humanoid.pth": ['0-CMU_85_85_12_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstand_failed2_subset1/Humanoid.pth": ['0-CMU_85_85_05_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_shape_pnn_handstand/Humanoid.pth": ['0-KIT_200_Handstand01_poses', '0-KIT_200_Handstand04_poses', '0-KIT_200_Handstand02_poses','0-KIT_200_Handstand05_poses', '0-KIT_200_Handstand06_poses', '0-Eyes_Japan_Dataset_aita_pose-16-handstand-aita_poses', '0-Eyes_Japan_Dataset_takiguchi_pose-16-handstand-takiguchi_poses', '0-Eyes_Japan_Dataset_shiono_pose-16-handstand-shiono_poses', '0-MPI_Limits_03099_op4_poses',  '0-MPI_Limits_03099_op5_poses'],

    # cartwheels
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset4/Humanoid.pth": ['0-Eyes_Japan_Dataset_hamada_turn-04-cartwheels-hamada_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset3/Humanoid_00048000.pth": ['0-BioMotionLab_NTroje_rub011_0030_scamper_poses'], 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset2/Humanoid.pth": ['0-BioMotionLab_NTroje_rub009_0030_scamper1_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset1/Humanoid.pth": ['0-ACCAD_Male2MartialArtsExtended_c3d_Extended 3_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed2_subset2/Humanoid.pth": ['0-Eyes_Japan_Dataset_takiguchi_turn-04-cartwheels-takiguchi_poses'], 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels/Humanoid.pth": ['0-BioMotionLab_NTroje_rub087_0030_scamper_poses', '0-CMU_88_88_07_poses'],

    #backflips
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_backflips_failed2_subset3/Humanoid.pth": ['0-CMU_88_88_05_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_backflips_failed2_subset2/Humanoid.pth": ['0-CMU_90_90_29_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_backflips_failed2_subset1/Humanoid.pth": ['0-CMU_85_85_14_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_backflips/Humanoid.pth": [ '0-CMU_90_90_19_poses', '0-CMU_90_90_28_poses', '0-CMU_88_88_08_poses', '0-CMU_88_88_09_poses', '0-CMU_90_90_08_poses', '0-CMU_85_85_01_poses', '0-CMU_85_85_02_poses'],

    # rolls 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_roll/Humanoid.pth": ['0-CMU_90_90_33_poses', '0-CMU_90_90_34_poses', '0-CMU_90_90_35_poses',  '0-CMU_90_90_36_poses',  '0-CMU_127_127_23_poses', '0-CMU_127_127_24_poses', '0-CMU_128_128_10_poses', '0-CMU_128_128_11_poses', '0-CMU_90_90_01_poses'],

    # scamper 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_scamper_failed3/Humanoid.pth": ['0-BioMotionLab_NTroje_rub048_0030_scamper_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_scamper_failed2/Humanoid.pth": ['0-BioMotionLab_NTroje_rub010_0029_scamper_poses'], 

    # weirld walks
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_weirdwalks/Humanoid.pth": ['0-BioMotionLab_NTroje_rub012_0031_scamper_poses'],

    # kicks punches 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_kickspunches/Humanoid.pth": ['0-ACCAD_Male2MartialArtsKicks_c3d_G7 -  capoera_poses'],

    # sprints
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_sprints/Humanoid.pth": ['0-ACCAD_Male2Running_c3d_C20 - run to pickup box_poses'],

    # floor_poses
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_floorposes/Humanoid.pth": ['0-ACCAD_Male1General_c3d_General A8 - Crouch to Lie Down_poses', '0-ACCAD_Male2General_c3d_A12- Crawl Backwards_poses'], 

    # front_flips 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_frontflips/Humanoid.pth": [ '0-CMU_85_85_06_poses', '0-CMU_85_85_07_poses'],

    # break_dance 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_breakdance/Humanoid.pth": ['0-CMU_85_85_08_poses', '0-CMU_85_85_04_poses'],


    # New stuff
    # ballet 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ballet/Humanoid.pth": ["0-CMU_05_05_06_poses", "0-CMU_05_05_17_poses", "0-CMU_05_05_18_poses"] , 


    # second_floor 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_second_floor/Humanoid.pth": ["0-SFU_0007_0007_Crawling001_poses","0-MPI_HDM05_dg_HDM_dg_03-09_01_120_poses"  ], 

    # second_kicks_punches
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_second_kicks_punches/Humanoid.pth": ["0-MPI_HDM05_tr_HDM_tr_05-01_03_120_poses"], 

    # second sprints 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_second_sprints/Humanoid.pth": [ "0-MPI_HDM05_dg_HDM_dg_05-03_02_120_poses"],

    # second roll 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_second_roll/Humanoid.pth": ["0-SFU_0017_0017_ParkourRoll001_poses"],

    # throw 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_throw/Humanoid.pth": ["0-MPI_HDM05_dg_HDM_dg_03-03_01_120_poses", "0-MPI_HDM05_tr_HDM_tr_03-03_03_120_poses", "0-MPI_HDM05_dg_HDM_dg_03-03_02_120_poses", "0-MPI_HDM05_mm_HDM_mm_03-03_01_120_poses", ] ,

    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_throw_failed1_subset1/Humanoid.pth":  ['0-MPI_HDM05_mm_HDM_mm_03-03_02_120_poses'], 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_throw_failed1_subset2/Humanoid.pth":  ['0-MPI_HDM05_tr_HDM_tr_03-03_01_120_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_throw_failed1_subset3/Humanoid.pth": ['0-MPI_HDM05_tr_HDM_tr_03-03_02_120_poses'], 

    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_limp/Humanoid.pth": ["0-MPI_HDM05_tr_HDM_tr_05-03_02_120_poses", "0-MPI_HDM05_dg_HDM_dg_05-03_01_120_poses", "0-MPI_HDM05_tr_HDM_tr_05-03_03_120_poses", "0-MPI_HDM05_tr_HDM_tr_05-03_01_120_poses", ], 


    # thirs stuff 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_thrid_sprint/Humanoid.pth" : ['0-MPI_HDM05_mm_HDM_mm_05-03_02_120_poses'],
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_third_kicks_punches/Humanoid.pth": ['0-MPI_HDM05_tr_HDM_tr_03-02_04_120_poses'],


    # last stuff 
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_cartwheel/Humanoid.pth" : ['0-CMU_49_49_08_poses'], # cartwheel # good!

    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_spinjump/Humanoid.pth" : ['0-CMU_87_87_01_poses'], # spin jump (use ballet) # good! 


    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_basketball/Humanoid.pth": ['0-MPI_HDM05_tr_HDM_tr_06-01_01_120_poses'], # basketball/volleyball poses and throws #use throw # good! 

    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_floor/Humanoid.pth" : ['0-TotalCapture_s4_freestyle1_poses'], # frog jumps, into crawl on the ground, into stand up (# use second_floor) # good




    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_throw/Humanoid.pth" : ['0-MPI_HDM05_dg_HDM_dg_03-03_03_120_poses'], # get up throw sit down throw #use throw # good : keep training 


    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed1_actual/Humanoid.pth": ['0-BMLmovi_Subject_47_F_MoSh_Subject_47_F_21_poses'], # "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed1_actual/Humanoid.pth"

    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed2/Humanoid.pth" : ['0-BMLmovi_Subject_3_F_MoSh_Subject_3_F_17_poses'],  # /move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed2/Humanoid.pth
    "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed3/Humanoid.pth" : ['0-BMLmovi_Subject_76_F_MoSh_Subject_76_F_20_poses'],  # "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed3/Humanoid.pth"


}



# SECOND COMPLETED (because we ned to train the "validation" set)
ballet =[
"0-CMU_05_05_06_poses", # ballet, spins into spin jump
"0-CMU_05_05_17_poses", # leap into backwards leap 
"0-CMU_05_05_18_poses"] # spin jump $ good! 

second_floor=[
"0-SFU_0007_0007_Crawling001_poses", # floor poses 
"0-MPI_HDM05_dg_HDM_dg_03-09_01_120_poses" # situps into pushups into pikes 
] # good!

second_kicks_punches =[
"0-MPI_HDM05_tr_HDM_tr_05-01_03_120_poses" # kicks and punches
] # good! 

second_sprints = [ "0-MPI_HDM05_dg_HDM_dg_05-03_02_120_poses"] # good! 

second_roll = [
"0-SFU_0017_0017_ParkourRoll001_poses"  # roll forwards # good! 
]

throw =[
"0-MPI_HDM05_dg_HDM_dg_03-03_01_120_poses", # sit down throw, stand up throw
"0-MPI_HDM05_tr_HDM_tr_03-03_03_120_poses", # sit down throw, stand up throw
"0-MPI_HDM05_dg_HDM_dg_03-03_02_120_poses", # sit down throw, stand up throw
"0-MPI_HDM05_mm_HDM_mm_03-03_01_120_poses", # sit down throw, stand up throw
# "0-MPI_HDM05_tr_HDM_tr_03-03_01_120_poses", # sit down throw, stand up throw
# "0-MPI_HDM05_tr_HDM_tr_03-03_02_120_poses", #  sit down throw, stand up throw
# "0-MPI_HDM05_mm_HDM_mm_03-03_02_120_poses" 
]# sit down throw, stand up throw

throw_failed1_subset1 = ['0-MPI_HDM05_mm_HDM_mm_03-03_02_120_poses'] # good! 
throw_failed1_subset2 = ['0-MPI_HDM05_tr_HDM_tr_03-03_01_120_poses'] # good! 
throw_failed1_subset3 = ['0-MPI_HDM05_tr_HDM_tr_03-03_02_120_poses'] # good! 
# Running 


limp =[
# "0-MPI_HDM05_tr_HDM_tr_05-03_04_120_poses", # limp into sprints into cartwheels 
"0-MPI_HDM05_tr_HDM_tr_05-03_02_120_poses",  # limp into sprints into cartwheels 
"0-MPI_HDM05_dg_HDM_dg_05-03_01_120_poses", # limping into sudden sprints into cart
"0-MPI_HDM05_tr_HDM_tr_05-03_03_120_poses", # limp into sprints into cartwheels 
"0-MPI_HDM05_tr_HDM_tr_05-03_01_120_poses", # limp into sprints into cartwheels 
 ] 

limp_failed1_subset1 = ["0-MPI_HDM05_tr_HDM_tr_05-03_04_120_poses"] # 



# THIRD COMPLETED (forgot some motions)
third_sprints = ['0-MPI_HDM05_mm_HDM_mm_05-03_02_120_poses']
third_kicks_punches = ['0-MPI_HDM05_tr_HDM_tr_03-02_04_120_poses']




# fourth layer (some still failed so adding them back) 
fourth_cartwheel = ['0-CMU_49_49_08_poses'] # cartwheel # good!
fourth_spinjump = ['0-CMU_87_87_01_poses'] # spin jump (use ballet) # good! 

fourth_basketball = ['0-MPI_HDM05_tr_HDM_tr_06-01_01_120_poses'] # basketball/volleyball poses and throws #use throw # good! 

fourth_floor = ['0-TotalCapture_s4_freestyle1_poses'] # frog jumps, into crawl on the ground, into stand up (# use second_floor) # good




fourth_throw = ['0-MPI_HDM05_dg_HDM_dg_03-03_03_120_poses'] # get up throw sit down throw #use throw # good : keep training 


fourth_sitdown =[
'0-BMLmovi_Subject_47_F_MoSh_Subject_47_F_21_poses', # sit down cross legged to get up  # use throw
'0-BMLmovi_Subject_3_F_MoSh_Subject_3_F_17_poses', # sit down cross legged to get up 
'0-BMLmovi_Subject_76_F_MoSh_Subject_76_F_20_poses'] # sit down cross legged to get up 
# all bad 

fourth_sitdown_failed1 = ['0-BMLmovi_Subject_47_F_MoSh_Subject_47_F_21_poses'] # "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed1_actual/Humanoid.pth"

fourth_sitdown_failed2 = ['0-BMLmovi_Subject_3_F_MoSh_Subject_3_F_17_poses'] # /move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed2/Humanoid.pthb
fourth_sitdown_failed3 = ['0-BMLmovi_Subject_76_F_MoSh_Subject_76_F_20_poses'] # "/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed3/Humanoid.pth"

# WAITING TO RUN 


class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
    # Assume each joint is 3dof
    diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

    return dof_vel[1:, :].flatten()


def compute_motion_dof_vels(motion):
    num_frames = motion.tensor.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

    return dof_vels

    
class DeviceCache:

    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out

class MotionlibMode(Enum):
    file = 1
    directory = 2
    
class MotionLibBase():

    def __init__(self, motion_file,  device, fix_height=FixHeightMode.full_fix, masterfoot_conifg=None, min_length=-1, im_eval=False, multi_thread=True,
    collect_start_idx=0, collect_step_idx=None,
    m2t_map_path=None, mode=None
    ):
        self._device = device
        self.mesh_parsers = None

        # MICHAEL===
        self.collect_start_idx = collect_start_idx
        self.collect_step_idx = collect_step_idx
        assert mode is not None
        self.mode_ = mode
        assert self.collect_step_idx is not None
        print(f'Collection start idx: {self.collect_start_idx}')
        if self.mode_ == 'eval':
            self.m2t_map_path = m2t_map_path
            self.m2t_map = np.load(self.m2t_map_path, allow_pickle=True)['motion_to_text_map'][()]
    
        # ===
        
        self.load_data(motion_file,  min_length = min_length, im_eval = im_eval)
        self.setup_constants(fix_height = fix_height, masterfoot_conifg = masterfoot_conifg, multi_thread = multi_thread)

        

        if flags.real_traj:
            if self._masterfoot_conifg is None:
                self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [13, 18, 23])   
            else:
                self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [19, 24, 29])
        return
        
    def load_data(self, motion_file,  min_length=-1, im_eval = False):
        if osp.isfile(motion_file): 
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:           
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        
        data_list = self._motion_data_load

        self.motion_to_len = {}
        for k, v in data_list.items():
            self.motion_to_len[k] = len(v['trans_orig'])

        if self.mode == MotionlibMode.file:
            if min_length != -1:
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
                # data_list = self._motion_data
            else:
                data_list = self._motion_data_load

            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 

    def setup_constants(self, fix_height = FixHeightMode.full_fix, masterfoot_conifg=None, multi_thread = True):
        self._masterfoot_conifg = masterfoot_conifg
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches
        self._sampling_batch_prob = None  # For use in sampling within batches
        
        
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, mesh_parsers, masterfoot_config, max_len, queue, pid):
        raise NotImplementedError

    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        raise NotImplementedError



    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1):
        def create_duplicated_list(n, num_duplicates, limit, start_idx, device):
            # Calculate the number of unique values needed
            num_unique_values = (n + num_duplicates - 1) // num_duplicates

            # Create a tensor of indices with each number repeated 'num_duplicates' times
            indices = torch.arange(start_idx, start_idx + num_unique_values).repeat_interleave(num_duplicates)

            # Apply the modulo operation to ensure values are within the limit
            sample_idxes = torch.remainder(indices, limit).to(device)

            # Truncate the tensor to the desired length
            return sample_idxes[:n]

        # load motion load the same number of motions as there are skeletons (humanoids)
        if "gts" in self.__dict__:
            del self.gts, self.grs, self.lrs, self.grvs, self.gravs, self.gavs, self.gvs, self.dvs,
            del self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa
            if flags.real_traj:
                del self.q_gts, self.q_grs, self.q_gavs, self.q_gvs

        motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_bodies = []
        self._motion_aa = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        torch.cuda.empty_cache()
        gc.collect()

        total_len = 0.0
        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)
        # TAKARA    
        # random_sample = True
        # start_idx = 20
        # if random_sample:
        #     sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        # else:           
        #     sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions ).to(self._device)

        # sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=False).to(self._device)

        # sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + 0, 1).to(self._device) # create sample idx for one motion
        # import ipdb; ipdb.set_trace()
        # sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=False).to(self._device)
        ### SAMPLE RANDOM KEYS
        # start_idx = 1 #1300 #1300 #5000
        # assert start_idx <= self._num_unique_motions, 'start_idx must be less than the number of unique motions'
        num_duplicates = 1
        sample_idxes = create_duplicated_list(n=len(skeleton_trees) , num_duplicates=num_duplicates, limit=int(self._num_unique_motions), start_idx=start_idx, device=self._device)


        # SAMPLE BASED ON NAME
        # import ipdb; ipdb.set_trace()
        names = ['0-KIT_8_WalkingStraightForwards03_poses' , '0-KIT_4_WalkingStraightBackwards04_poses','0-KIT_9_WalkInClockwiseCircle05_poses', '0-KIT_10_WalkInCounterClockwiseCircle06_poses']
        # names = ['0-KIT_167_upstairs08_poses']#, '0-KIT_425_walking_01_poses', '0-KIT_183_walking_medium03_poses', '0-KIT_317_bend_right03_poses', '0-KIT_7_WalkInCounterClockwiseCircle05_poses', '0-KIT_205_walking_slow03_poses', '0-KIT_359_bend_right08_poses', '0-KIT_3_bend_left03_poses', '0-KIT_348_walking_fast02_poses', '0-KIT_424_walking_slow04_poses', '0-KIT_11_WalkInClockwiseCircle06_poses', '0-KIT_3_turn_left08_poses', '0-KIT_9_walking_slow02_poses', '0-KIT_425_bend_left05_poses', '0-KIT_3_walk_with_support03_poses', '0-KIT_12_WalkInCounterClockwiseCircle08_poses', '0-KIT_11_WalkingStraightForwards04_poses', '0-KIT_7_WalkingStraightForwards06_poses', '0-KIT_317_walking_slow03_poses', '0-KIT_3_bend_left01_poses', '0-KIT_3_walking_forward_4steps_right_05_poses', '0-KIT_12_WalkingStraightBackwards08_poses', '0-KIT_205_turn_left03_poses', '0-KIT_317_walking_fast04_poses', '0-KIT_6_LeftTurn02_1_poses', '0-KIT_167_walking_run05_poses', '0-KIT_8_WalkInClockwiseCircle06_poses', '0-KIT_11_WalkingStraightForwards07_poses', '0-KIT_317_bend_right06_poses', '0-KIT_4_WalkInClockwiseCircle03_poses', '0-KIT_3_turn_right10_poses', '0-KIT_3_bend_left06_poses', '0-KIT_6_WalkInCounterClockwiseCircle09_1_poses', '0-KIT_424_walking_slow07_poses', '0-KIT_317_bend_right01_poses', '0-KIT_7_WalkingStraightForwards04_poses', '0-KIT_9_LeftTurn10_poses', '0-KIT_7_WalkingStraightForwards09_poses', '0-KIT_183_walking_fast10_poses', '0-KIT_317_bend_left08_poses', '0-KIT_359_turn_left05_poses', '0-KIT_183_bend_left09_poses', '0-KIT_314_turn_left05_poses', '0-KIT_183_bend_right03_poses', '0-KIT_167_turn_right09_poses', '0-KIT_425_walking_05_poses', '0-KIT_167_turn_left08_poses', '0-KIT_167_turn_left02_poses', '0-KIT_425_walking_07_poses', '0-KIT_317_walking_fast07_poses', '0-KIT_7_WalkInClockwiseCircle10_poses', '0-KIT_317_walking_run07_poses', '0-KIT_9_WalkInCounterClockwiseCircle05_poses', '0-KIT_314_turn_left10_poses', '0-KIT_8_WalkInCounterClockwiseCircle03_poses', '0-KIT_359_walking_fast04_poses', '0-KIT_314_bend_left09_poses', '0-KIT_11_WalkingStraightBackwards07_1_poses', '0-KIT_167_bend_left01_poses', '0-KIT_9_walking_slow09_poses', '0-KIT_3_turn_left01_poses', '0-KIT_8_WalkingStraightForwards06_poses', '0-KIT_3_bend_left07_poses', '0-KIT_425_walking_medium08_poses', '0-KIT_183_walking_fast01_poses', '0-KIT_425_walking_slow02_poses', '0-KIT_424_turn_right09_poses', '0-KIT_8_RightTurn04_poses', '0-KIT_9_walking_slow08_poses', '0-KIT_11_WalkingStraightForwards01_poses', '0-KIT_317_walking_medium03_poses', '0-KIT_10_WalkInClockwiseCircle08_poses', '0-KIT_317_walking_fast02_poses', '0-KIT_8_WalkInClockwiseCircle02_poses', '0-KIT_9_LeftTurn01_poses', '0-KIT_11_WalkInCounterClockwiseCircle05_poses', '0-KIT_183_bend_left04_poses', '0-KIT_314_walking_run03_poses', '0-KIT_348_bend_right08_poses', '0-KIT_11_WalkInClockwiseCircle02_poses', '0-KIT_12_RightTurn07_poses', '0-KIT_317_bend_right05_poses', '0-KIT_3_walking_run06_poses', '0-KIT_424_bend_right05_poses', '0-KIT_8_WalkingStraightBackwards05_poses', '0-KIT_425_walking_slow11_poses', '0-KIT_317_turn_left10_poses', '0-KIT_348_bend_left01_poses', '0-KIT_314_walking_slow05_poses', '0-KIT_4_WalkingStraightBackwards03_poses', '0-KIT_424_bend_left08_poses', '0-KIT_3_walking_run05_poses', '0-KIT_8_WalkInClockwiseCircle09_poses', '0-KIT_6_RightTurn10_1_poses', '0-KIT_11_RightTurn06_poses', '0-KIT_314_turn_right11_poses', '0-KIT_9_bend_left04_poses', '0-KIT_348_walking_fast01_poses', '0-KIT_6_WalkInCounterClockwiseCircle01_1_poses', '0-KIT_424_bend_right08_poses']
        # names = ['0-KIT_1347_Experiment3_subject1347_wash_leg_position_smallcircles_02_poses']
        # names = ['0-KIT_200_Handstand04_poses']#, '0-KIT_200_Handstand06_poses', '0-KIT_200_Handstand02_poses', '0-KIT_200_Handstand01_poses', '0-KIT_200_Handstand05_poses']
        # names = ['0-ACCAD_Male2Walking_c3d_B18 -  Walk to leap to walk t2_poses']
        # names = ['0-BioMotionLab_NTroje_rub011_0030_scamper_poses']
        # names = ['0-MPI_HDM05_tr_HDM_tr_05-03_04_120_poses']
        #################################################################################################
        name2idx = {name: idx for idx, name in enumerate(self._motion_data_keys)}
        # names = [handstands[5]] #[break_dance[0]] #
        # names = ground2getup_failed3_subset1 #cartwheels_failed3_subset2  #handstands_failed3_subset2  #+ sprints + kicks_punches + floor_poses + weird_walks + break_dance + front_flips
        # names = cartwheels_failed3_subset3  #cartwheels_failed3_subset2  #handstands_failed3_subset2  #+ sprints + kicks_punches + floor_poses + weird_walks + break_dance + front_flips
        # names = handstands_failed3_subset1 #cartwheels_failed3_subset2  #handstands_failed3_subset2  #+ sprints + kicks_punches + floor_poses + weird_walks + break_dance + front_flips
        # names=ballet   
        # names=throw_failed1_subset3
        # names = second_kicks_punches
        # names = break_dance#[0] # + front_flips
        # names = [break_dance[0]]#[0] # + front_flips

        # clip_motion = False
        # names = limp_failed1_subset1
        # names= handstands_failed3_subset1
        # names= third_kicks_punches
        # names = fourth_basketball 
        # names = cartwheels_failed3_subset3
        # names = [names[1]]
        # names = [back_flips[6]]
        # names = limp_failed1_subset1
        names = second_kicks_punches
        # names = cartwheels_failed3_subset3
        # names = handstands_failed2_subset2
        # names = throw_failed1_subset1
        #names = FAILED_MOT_1 # 
        # print(len(final_failed))
        # names = [CARTWHEEL[0]]        
        # names = [FLIPS[0]]            
        # names = ['0-KIT_200_Handstand04_poses']
        # names = FAILED_MOT_1
        # names = FLIPS
        # len(names)      
    
        # names = ['0-SSM_synced_20160930_50032_punch_kick_sync_poses']
        # names =['0-Transitions_mocap_mazen_c3d_walksideways_running_poses']
        # names = [   '0-Transitions_mocap_mazen_c3d_devishdance_kick_poses',
        #             '0-Transitions_mocap_mazen_c3d_sit_jumpinplace_poses',
        #             '0-Transitions_mocap_mazen_c3d_walksideways_running_poses',
        #             '0-Transitions_mocap_mazen_c3d_walksideways_runbackwards_poses',
        #             '0-SSM_synced_20161014_50033_dribble_kick_sync_poses',
        #             '0-SSM_synced_20161014_50033_punch_kick_sync_poses',
        #             '0-SSM_synced_20160930_50032_punch_kick_sync_poses',  
        #             ] 
        # names = [   '0-Transitions_mocap_mazen_c3d_devishdance_kick_poses' ]

        sample_idxes = torch.tensor([name2idx[name] for name in names], device=self._device)
        sample_idxes = torch.sort(sample_idxes).values.repeat(550)
        np.random.shuffle(sample_idxes)     
        sample_idxes= sample_idxes[:len(skeleton_trees)]

        # #################################################################################################
        
        # SAMPLE BASED ON FILE NAMES: 
        # import ipdb; ipdb.set_trace()

        # all_names = np.load('walking_motion_names.npy')
        # name2idx ={}
        # for name in all_names: 
        #     # print(name)
        #     # import ipdb; ipdb.set_trace()
        #     if len(np.where(self._motion_data_keys == '0-'+name)[0]) != 0:
        #         name2idx[name] = np.where(self._motion_data_keys == '0-'+name)[0][0]
        #     else:
        #         # print(f"Motion {name} not found in motion data keys in motion_lib_base.py")
        #         pass
        # # import ipdb; ipdb.set_trace()   
        # np.random.seed(0)
        
        #############################################################################
        # if self.mode_ == 'eval':
        #     print(f'EVAL MODE: Building map to motion_names!')
        #     eval_idxs = []
        #     for name in self.m2t_map.keys():
        #         idx = np.where(self._motion_data_keys == '0-'+name)[0][0]
        #         eval_idxs.append(idx)
            
        #     eval_idxs = np.flip(np.array(sorted(eval_idxs)), axis=0)
        #     print(f'Number of eval keys: {len(eval_idxs)}')
            

        # def get_upsample_dist(m2a_map, action_counts, subset):
        #     action_labels = []
        #     for motion_name in subset:
        #         action_labels.append(m2a_map[motion_name[2:]])

        #     weights = np.array([1/action_counts[act] for act in action_labels])
        #     weights /= sum(weights)
        #     assert len(weights) == len(subset)
        #     return weights



        # ============
        # NOTE: For upsampled motions
        UPSAMPLE = False
        # if self.mode_ == 'collect' and UPSAMPLE:
        #     short_subset_mask = [self.motion_to_len[name] <= 800 for name in self._motion_data_keys]
        #     subset = self._motion_data_keys[short_subset_mask]
        #     m2a_map = np.load('motion_to_action_map_KIT.npz', allow_pickle=True)['motion_to_action_map'][()]
        #     action_counts = np.load('action_counts_KIT.npz', allow_pickle=True)['action_counts'][()]
        #     weights = get_upsample_dist(m2a_map, action_counts, subset)
        #     print(f'Number of short motions: {len(subset)}')
        # ============


        # ============
        # # NOTE: For retrying failed motions 
        # COLLECT_FAILED = False
        # failed_motion_file = 'collected_data/obs-phc_sigma=0.06/failed.txt'
        # if self.mode_ == 'collect' and COLLECT_FAILED:
        #     failed_idxs = []
        #     with open(failed_motion_file, 'r') as f:
        #         failed_names = f.readlines()
        #         for name in failed_names:
        #             idx = np.where(self._motion_data_keys == name.strip())[0][0]
        #             failed_idxs.append(idx)
                    
        #     failed_idxs = np.flip(np.array(sorted(failed_idxs)), axis=0)
        #     print(f'Number of failed keys: {len(failed_idxs)}')
        # # ============

        # print(f'Number of motion keys: {len(self._motion_data_keys)}')

        # # ===== Choose sample_idxes based on mode =====
        # start_idx = self.collect_start_idx
        # end_idx = min(start_idx + self.collect_step_idx, self._num_unique_motions )
        # sample_idxes = np.arange(start_idx, end_idx)
        # sample_idxes = failed_idxs[sample_idxes]
        
        # if self.mode_ == 'eval':
        #     end_idx = min(start_idx + self.collect_step_idx, len(eval_idxs))
        #     sample_idxes = np.arange(start_idx, end_idx)
        #     sample_idxes = eval_idxs[sample_idxes]
        # elif self.mode_ == 'collect' and COLLECT_FAILED:
        #     end_idx = min(start_idx + self.collect_step_idx, len(failed_idxs))
        #     sample_idxes = np.arange(start_idx, end_idx)
        #     sample_idxes = failed_idxs[sample_idxes]
        # else:
        #     pass
        #     assert False
        
        #     if UPSAMPLE:
        #         base_idxs = np.arange(len(self._motion_data_keys))
        #         base_idxs = base_idxs[short_subset_mask]
        #         upsample_idxs = np.random.choice(
        #             base_idxs, size=len(base_idxs),
        #             replace=True, p=weights
        #         )
        #         end_idx = min(start_idx + self.collect_step_idx, len(subset))
        #         sample_idxes = np.arange(start_idx, end_idx)
        #         sample_idxes = upsample_idxs[sample_idxes]
        #     else:
        #         end_idx = min(start_idx + self.collect_step_idx, len(self._motion_data_keys))
        #         sample_idxes = np.arange(start_idx, end_idx)
        # =====

        # sample_idxes = shuffled_idxs[sample_idxes] # permuation of motion data keys
        # sample_idxes = torch.tensor(sample_idxes, dtype=torch.long, device=self._device)

        # len(skeleton_trees) is the number of environments. Need to cut it here.
        # sample_idxes = sample_idxes[:len(skeleton_trees)]

        # np.random.shuffle(sample_idxes)
        # sample_idxes = torch.tensor(sample_idxes[:min(num_motions, len(skeleton_trees))], device=self._device)
        # sample_idxes = torch.tensor(sample_idxes, device=self._device)
        # sample_idxes = torch.sort(sample_idxes).values
        # sample_idxes = sample_idxes.repeat_interleave(num_duplicates)


        # start_idx = self.collect_start_idx
        # end_idx = min(start_idx + self.collect_step_idx, self._num_unique_motions )
        # sample_idxes = np.arange(start_idx, end_idx)
        # # sample_idxes = torch.tensor(sample_idxes, dtype=torch.long, device=self._device)
        # sample_idxes = sample_idxes[:len(skeleton_trees)]
        
        #######################################################################################3

        if isinstance(sample_idxes, np.ndarray):
            sample_idxes = torch.tensor(sample_idxes, device=self._device)
        
        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(self._curr_motion_ids, num_classes = self._num_unique_motions).to(self._device)  # Testing for obs_v5
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        print("\n****************************** Current motion keys ******************************")
        
        # print("Sampling motion:", sample_idxes[:50])
        print(self.curr_motion_keys)
        # if len(self.curr_motion_keys) < 100:
        #     print(self.curr_motion_keys)
        # else:       
        #     print(self.curr_motion_keys[:50], ".....")
        print("*********************************************************************************\n")

        print(f'Cur motion ids: {self._curr_motion_ids}')

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        mp.set_sharing_strategy('file_descriptor')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(mp.cpu_count(), 64)

        if num_jobs <= 8 or not self.multi_thread:
            num_jobs = 1
        if flags.debug:
            num_jobs = 1
        
        res_acc = {}  # using dictionary ensures order of the results.
        jobs = motion_data_list
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [(ids[i:i + chunk], jobs[i:i + chunk], skeleton_trees[i:i + chunk], gender_betas[i:i + chunk], self.fix_height, self.mesh_parsers, self._masterfoot_conifg, max_len) for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]
        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

        for i in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            res_acc.update(res)

        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            # # TAkara
            # if clip_motion: 
            #     start= 1200 
            #     orig_len = motion_file_data['pose_quat_global'].shape[0]

            #     motion_file_data['pose_quat_global'] = motion_file_data['pose_quat_global'][start:]

            #     motion_file_data['pose_quat'] = motion_file_data['pose_quat'][start:]
            #     motion_file_data['trans_orig'] = motion_file_data['trans_orig'][start:]
            #     motion_file_data['root_trans_offset'] = motion_file_data['root_trans_offset'][start:]
            #     motion_file_data['pose_aa'] = motion_file_data['pose_aa'][start:]

                # curr_motion.global_rotation= curr_motion.global_rotation[start:]

                # for attr_name in dir(curr_motion):
                #     # Skip special and private attributes
                #     if attr_name.startswith('__'):
                #         continue
                    
                #     try:
                #         attr_value = getattr(curr_motion, attr_name, None)

                #         # Check if the attribute is a PyTorch tensor and its shape
                #         if isinstance(attr_value, torch.Tensor) and attr_value.shape[0] == orig_len:
                #             # import ipdb; ipdb.set_trace()

                #             # Assuming you want to clip the tensor from index 'start' to the end
                #             # Ensure 'start' is within the tensor's range to avoid 'index out of range' errors
                #             if start < attr_value.shape[0]:
                #                 # Clip the tensor
                #                 clipped_tensor = attr_value[start:]
                #                 # Optionally update the attribute with the clipped tensor if needed
                #                 setattr(curr_motion, attr_name, clipped_tensor)
                #             else:
                #                 print(f"Skipping {attr_name} because start index {start} is out of range.")

                #     except AttributeError as e:
                #         # Handle the case where the attribute can't be set
                #         print(f"Cannot set attribute '{attr_name}': {e}")
    
                # import ipdb; ipdb.set_trace()
                # import ipdb; ipdb.set_trace()

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            # if clip_motion:
            #     num_frames = motion_file_data['pose_quat_global'].shape[0]
            # else:
            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            
            if "beta" in motion_file_data:
                self._motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                self._motion_bodies.append(curr_motion.gender_beta)
            else:
                self._motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                self._motion_bodies.append(torch.zeros(17))

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion
            
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(self._motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(self._motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
            self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
            self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
            self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = motion.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    # def update_sampling_weight(self):
    #     ## sampling weight based on success rate. 
    #     # sampling_temp = 0.2
    #     sampling_temp = 0.1
    #     curr_termination_prob = 0.5

    #     curr_succ_rate = 1 - self._termination_history[self._curr_motion_ids] / self._sampling_history[self._curr_motion_ids]
    #     self._success_rate[self._curr_motion_ids] = curr_succ_rate
    #     sample_prob = torch.exp(-self._success_rate / sampling_temp)

    #     self._sampling_prob = sample_prob / sample_prob.sum()
    #     self._termination_history[self._curr_motion_ids] = 0
    #     self._sampling_history[self._curr_motion_ids] = 0

    #     topk_sampled = self._sampling_prob.topk(50)
    #     print("Current most sampled", self._motion_data_keys[topk_sampled.indices.cpu().numpy()])
        
    def update_hard_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only trained on "failed" sequences. Auto PMCP. 
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._sampling_prob[:] = 0
            self._sampling_prob[indexes] = 1/len(indexes)
            print("############################################################ Auto PMCP ############################################################")
            print(f"Training on only {len(failed_keys)} seqs")
            print(failed_keys)
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches
            
    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP. 
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)    
            
            print("############################################################ Auto PMCP ############################################################")
            print(f"Training mostly on {len(self._sampling_prob.nonzero())} seqs ")
            print(self._motion_data_keys[self._sampling_prob.nonzero()].flatten())
            print(f"###############################################################################################################################")
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

    def update_sampling_prob(self, termination_history):
        if len(termination_history) == len(self._termination_history):
            self._sampling_prob[:] = termination_history/termination_history.sum()
            self._termination_history = termination_history
            return True
        else:
            return False
        
        
    # def update_sampling_history(self, env_ids):
    #     self._sampling_history[self._curr_motion_ids[env_ids]] += 1
    #     # print("sampling history: ", self._sampling_history[self._curr_motion_ids])

    # def update_termination_history(self, termination):
    #     self._termination_history[self._curr_motion_ids] += termination
    #     # print("termination history: ", self._termination_history[self._curr_motion_ids])

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob, num_samples=n, replacement=True).to(self._device)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_interval(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time
        curr_fps = 1 / 30
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps

        return motion_time

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * 30 / self._motion_fps).int()
        else:
            return (self._motion_num_frames[motion_ids] * 30 / self._motion_fps).int()

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        # import ipdb; ipdb.set_trace()
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1


        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = torch_utils.slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel

        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        }

    def get_root_pos_smpl(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        vals = [rg_pos0, rg_pos1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        return {"root_pos": rg_pos[..., 0, :].clone()}

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)